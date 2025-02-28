#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

void convertCppToAssembly(const std::string& inputFile, const std::string& outputFile) {
    // Command to compile C++ code to assembly
    std::string command = "g++ -S -o " + outputFile + " " + inputFile;

    // Execute the command
    int result = std::system(command.c_str());
    
    if (result != 0) {
        std::cerr << "Error: Failed to convert C++ to Assembly." << std::endl;
    } else {
        std::cout << "Successfully converted C++ to Assembly. Output written to " << outputFile << std::endl;
    }
}

int main() {
    // Specify the input and output files
    std::string inputFile = "dfs_code.cpp"; // Input file containing C++ DFS code
    std::string outputFile = "dfs_code.s"; // Output file for assembly code

    // Convert the C++ code to assembly
    convertCppToAssembly(inputFile, outputFile);

    return 0;
}
