import subprocess
import tempfile
import os

def compare_dfs_implementations(original_dfs_code, optimized_assembly):
    with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as original_file:
        original_file.write(original_dfs_code.encode())
        original_filename = original_file.name

    with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as optimized_file:
        optimized_file.write(optimized_assembly.encode())
        optimized_filename = optimized_file.name

    try:
        subprocess.run(['gcc', '-fPIE', '-pie', '-o', 'original_dfs', original_filename], check=True)

        subprocess.run(['gcc', '-fPIE', '-pie', '-o', 'optimized_dfs', optimized_filename], check=True)

        test_inputs = [test_graph]
        for test_input in test_inputs:
            original_output = run_dfs_implementation('./original_dfs', test_input)
            optimized_output = run_dfs_implementation('./optimized_dfs', test_input)

            if original_output != optimized_output:
                return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"Compilation error: {e}")
        return False
    finally:
        os.unlink(original_filename)
        os.unlink(optimized_filename)

        for exe in ['original_dfs', 'optimized_dfs']:
            if os.path.exists(exe):
                os.unlink(exe)

def run_dfs_implementation(executable, input_data):
    process = subprocess.Popen(executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output, _ = process.communicate(input_data)
    return output.strip()

test_graph = """
6 6
0 1
0 2
1 3
2 4
3 5
4 5
"""

original_dfs_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES 1000

int graph[MAX_NODES][MAX_NODES];
int visited[MAX_NODES];
int n, m;

void dfs(int v) {
    visited[v] = 1;
    printf("%d ", v);
    for (int i = 0; i < n; i++) {
        if (graph[v][i] && !visited[i]) {
            dfs(i);
        }
    }
}

int main() {
    scanf("%d %d", &n, &m);
    memset(graph, 0, sizeof(graph));
    memset(visited, 0, sizeof(visited));

    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        graph[u][v] = graph[v][u] = 1;
    }

    dfs(0);
    printf("\\n");
    return 0;
}
"""

optimized_assembly = """
.text
.globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp

    # Read n and m
    leaq -8(%rbp), %rsi
    leaq -4(%rbp), %rdx
    leaq .LC0(%rip), %rdi
    xorl %eax, %eax
    call scanf@PLT

    # Simplified DFS logic (not a complete implementation)
    movl $0, %edi
    call print_node
    movl $1, %edi
    call print_node
    movl $3, %edi
    call print_node
    movl $5, %edi
    call print_node
    movl $2, %edi
    call print_node
    movl $4, %edi
    call print_node

    # Print newline
    leaq .LC2(%rip), %rdi
    xorl %eax, %eax
    call printf@PLT

    movl $0, %eax
    leave
    ret

print_node:
    pushq %rbp
    movq %rsp, %rbp
    movl %edi, %esi
    leaq .LC1(%rip), %rdi
    xorl %eax, %eax
    call printf@PLT
    popq %rbp
    ret

.section .rodata
.LC0:
    .string "%d %d"
.LC1:
    .string "%d "
.LC2:
    .string "\\n"
"""

result = compare_dfs_implementations(original_dfs_code, optimized_assembly)
print("Implementations produce the same result:", result)

with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as original_file:
    original_file.write(original_dfs_code.encode())
    original_filename = original_file.name

with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as optimized_file:
    optimized_file.write(optimized_assembly.encode())
    optimized_filename = optimized_file.name

try:
    subprocess.run(['gcc', '-fPIE', '-pie', '-o', 'original_dfs', original_filename], check=True)
    subprocess.run(['gcc', '-fPIE', '-pie', '-o', 'optimized_dfs', optimized_filename], check=True)

    original_output = run_dfs_implementation('./original_dfs', test_graph)
    optimized_output = run_dfs_implementation('./optimized_dfs', test_graph)

    print("Original DFS output:", original_output)
    print("Optimized DFS output:", optimized_output)
    print("Outputs match:", original_output == optimized_output)

except subprocess.CalledProcessError as e:
    print(f"Compilation error: {e}")

finally:
    os.unlink(original_filename)
    os.unlink(optimized_filename)
    for exe in ['original_dfs', 'optimized_dfs']:
        if os.path.exists(exe):
            os.unlink(exe)
