#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh/IO/OFF.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <filesystem>
#include <map>
#include <string>

#include "cnpy.h"

namespace fs = std::filesystem;
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
namespace SMS = CGAL::Surface_mesh_simplification;

// ---------------------------
// Read DEM from .txt
// ---------------------------
bool read_dem_txt(const std::string& filename, Surface_mesh& mesh) {
    std::ifstream infile(filename);
    if (!infile) return false;

    int cols, rows;
    if (!(infile >> cols >> rows)) return false;

    std::vector<std::vector<double>> elevations(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (!(infile >> elevations[i][j])) return false;

    mesh.clear();
    std::vector<std::vector<Surface_mesh::Vertex_index>> vertex_grid(rows, std::vector<Surface_mesh::Vertex_index>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            vertex_grid[i][j] = mesh.add_vertex(Point_3(j, i, elevations[i][j]));

    for (int i = 0; i < rows - 1; ++i)
        for (int j = 0; j < cols - 1; ++j) {
            mesh.add_face(vertex_grid[i][j], vertex_grid[i + 1][j], vertex_grid[i][j + 1]);
            mesh.add_face(vertex_grid[i + 1][j], vertex_grid[i + 1][j + 1], vertex_grid[i][j + 1]);
        }

    return true;
}

// ---------------------------
// Read DEM from .npy
// ---------------------------
bool read_dem_npy(const std::string& filename, Surface_mesh& mesh) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    if (arr.word_size != sizeof(double)) return false; 
    if (arr.shape.size() != 2) return false;

    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];
    double* data = arr.data<double>();

    mesh.clear();
    std::vector<std::vector<Surface_mesh::Vertex_index>> vertex_grid(rows, std::vector<Surface_mesh::Vertex_index>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            vertex_grid[i][j] = mesh.add_vertex(Point_3(j, i, data[i*cols + j]));

    for (size_t i = 0; i < rows - 1; ++i)
        for (size_t j = 0; j < cols - 1; ++j) {
            mesh.add_face(vertex_grid[i][j], vertex_grid[i + 1][j], vertex_grid[i][j + 1]);
            mesh.add_face(vertex_grid[i + 1][j], vertex_grid[i + 1][j + 1], vertex_grid[i][j + 1]);
        }

    return true;
}

// ---------------------------
// Save OBJ
// ---------------------------
void save_mesh_obj(const std::string& filename, const Surface_mesh& mesh) {
    std::ofstream out(filename);
    std::map<Surface_mesh::Vertex_index, int> vertex_map;
    int vertex_count = 1;

    for (auto v : mesh.vertices()) {
        Point_3 p = mesh.point(v);
        out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
        vertex_map[v] = vertex_count++;
    }

    for (auto f : mesh.faces()) {
        out << "f";
        for (auto v : mesh.vertices_around_face(mesh.halfedge(f))) {
            out << " " << vertex_map[v];
        }
        out << "\n";
    }
}

// ---------------------------
// Simplify
// ---------------------------
void simplify_and_save(const Surface_mesh& original_mesh, const std::string& dataset_name) {
    std::vector<double> retention_rates = {0.80, 0.60, 0.40, 0.20, 0.10, 0.01};
    fs::create_directory("simplified");

    for (double retention : retention_rates) {
        Surface_mesh mesh = original_mesh;
        size_t initial_vertices = mesh.number_of_vertices();
        size_t target_vertices = std::max((size_t)(initial_vertices * retention), size_t(10));

        auto stop = [&mesh, target_vertices](double, const SMS::Edge_profile<Surface_mesh>&,
                                             std::size_t, std::size_t) -> bool {
            return mesh.number_of_vertices() <= target_vertices;
        };

        int edges_removed = SMS::edge_collapse(mesh, stop);

        std::stringstream ss_obj, ss_off;
        ss_obj << "simplified/" << dataset_name << "_simplified_" << (int)(retention*100) << "pct.obj";
        ss_off << "simplified/" << dataset_name << "_simplified_" << (int)(retention*100) << "pct.off";

        save_mesh_obj(ss_obj.str(), mesh);

        std::ofstream off_file(ss_off.str());
        CGAL::write_off(off_file, mesh);

        std::cout << dataset_name << ": " << (retention*100) << "% → "
                  << mesh.number_of_vertices() << " vertices, edges removed " << edges_removed << std::endl;
    }
}

// ---------------------------
// Main
// ---------------------------
int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "data";
    fs::path file_path(path);

    if (!fs::exists(file_path)) {
        std::cerr << "❌ File not found: " << path << std::endl;
        return EXIT_FAILURE;
    }

    Surface_mesh mesh;
    bool success = false;

    if (file_path.extension() == ".txt") success = read_dem_txt(path, mesh);
    else if (file_path.extension() == ".npy") success = read_dem_npy(path, mesh);
    else {
        std::cerr << "⚠️ Unsupported file: " << path << std::endl;
        return EXIT_FAILURE;
    }

    if (!success) {
        std::cerr << "❌ Failed to read: " << path << std::endl;
        return EXIT_FAILURE;
    }

    if (!CGAL::is_triangle_mesh(mesh)) {
        std::cerr << "❌ Not a valid triangle mesh: " << path << std::endl;
        return EXIT_FAILURE;
    }

    std::string dataset_name = file_path.stem().string();
    simplify_and_save(mesh, dataset_name);

    std::cout << "✅ Finished " << dataset_name << std::endl;
    return EXIT_SUCCESS;
}
