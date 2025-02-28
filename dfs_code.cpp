#include <iostream>
#include <list>
using namespace std;

class Graph {
    int V; // Number of vertices
    list<int>* adj; // Pointer to an array containing adjacency lists

public:
    Graph(int V); // Constructor
    void addEdge(int v, int w); // Function to add an edge
    void DFS(int v); // Function that performs DFS
private:
    void DFSUtil(int v, bool visited[]); // Utility function for DFS
};

Graph::Graph(int V) {
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w); // Add w to v’s list
}

void Graph::DFSUtil(int v, bool visited[]) {
    visited[v] = true;
    cout << v << " ";

    for (auto i = adj[v].begin(); i != adj[v].end(); ++i) {
        if (!visited[*i]) {
            DFSUtil(*i, visited);
        }
    }
}

void Graph::DFS(int v) {
    bool* visited = new bool[V];
    for (int i = 0; i < V; i++) {
        visited[i] = false;
    }
    
    DFSUtil(v, visited);
}

int main() {
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    
    cout << "Depth First Traversal starting from vertex 2:\n";
    g.DFS(2);
    
    return 0;
}
