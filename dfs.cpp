#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <bits/stdc++.h>

using namespace std;

vector<vector<int>> adj; // graph represented as an adjacency list
int n; // number of vertices

vector<bool> visited;

// Recursive dfs
void recursive_dfs(int v) {
    visited[v] = true;
    for (int u : adj[v]) {
        if (!visited[u])
            recursive_dfs(u);
    }
}

// Iterative dfs
void iterative_dfs(int v) {
    visited[v] = true;
    stack<int> st;
    st.push(v);
    while(!st.empty()) {
        int u = st.top();
        st.pop();
        for (int w : adj[u]){
            if (!visited[w]){
                visited[w] = true;
                st.push(w);
            }
        }
    }
}

vector<vector<int>> build_adj_list(const string& file_path, int& num_nodes){

    ifstream file(file_path);

    if (!file) {
        cerr << "File path not found\n";
    }

    // Skip comment lines and parse metadata
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            if (line.find("Nodes:") != string::npos) {
                istringstream iss(line);
                string temp;
                iss >> temp >> temp >> num_nodes; // Extract the number after "Nodes:"
            }
            continue;
        }

        break; // Stop skipping lines once we reach actual data
    }

    vector<vector<int>> adjacency_list(num_nodes);

    do {
        istringstream iss(line);
        int from_node, to_node;
        if (iss >> from_node >> to_node) {
            // Add edge from 'from_node' to 'to_node'
            adjacency_list[from_node].push_back(to_node);

            // Add edge from 'to_node' to 'from_node' (since the graph is undirected)
            adjacency_list[to_node].push_back(from_node);
        }
    } while (getline(file, line));

    return adjacency_list;
    
}

int main(){
    // read n and adj from input
    int m,u,v;
    cin >> n >> m;
    adj.resize(n);

    auto start = chrono::high_resolution_clock::now();

    for(int i = 0; i < m; i++){
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Duration of taking the input
    cout<< "Time taken to take input from the graph: " << duration.count() << " seconds\n";

    visited.assign(n, false);


    start = chrono::high_resolution_clock::now();

    for (int v = 0; v < n; v++) {
        if (!visited[v])
            recursive_dfs(v);
    }

    end = chrono::high_resolution_clock::now();
    
    duration = end - start;

    // Duration of recursive dfs
    cout << "Time taken to run recursive dfs: " << duration.count() << " seconds\n";

    
    visited.assign(n, false);

    start = chrono::high_resolution_clock::now();

    for (int v = 0; v < n; v++){
        if (!visited[v])
            iterative_dfs(v);
    }

    end = chrono::high_resolution_clock::now();
    
    cout << "Time taken to run iterative dfs: " << duration.count() << " seconds\n";

}