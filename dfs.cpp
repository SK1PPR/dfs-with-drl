#include <iostream>
#include <vector>
#include <chrono>

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