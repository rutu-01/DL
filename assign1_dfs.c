/* Design and implement Parallel Breadth First Search and Depth First Search based on existing
algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS .*/

#include <stdio.h>
#include <omp.h>

#define MAX_VERTICES 100

int visited[MAX_VERTICES];
int graph[MAX_VERTICES][MAX_VERTICES];

void DFS(int graph[MAX_VERTICES][MAX_VERTICES], int v) {
  visited[v] = 1;
  printf("%d ", v);

  #pragma omp task firstprivate(v) // Avoid race conditions on visited flag
  for (int i = 0; i < MAX_VERTICES; i++) {
    if (graph[v][i] && !visited[i]) {
      DFS(graph, i);
    }
  }
}

int main() {
  int V, E, i, j;

  printf("Enter the number of vertices: ");
  scanf("%d", &V);

  printf("Enter the number of edges: ");
  scanf("%d", &E);

  printf("Enter the graph connections (source, destination):\n");
  for (i = 0; i < E; i++) {
    scanf("%d %d", &j, &j);
    graph[i][j] = graph[j][i] = 1;
  }

  printf("Enter the starting vertex for DFS: ");
  scanf("%d", &i);

  #pragma omp parallel
  DFS(graph, i);

  printf("\n");
  return 0;
}
