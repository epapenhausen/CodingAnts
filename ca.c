#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>

#include <isl/aff.h>
#include <isl/ast.h>

#include "schedule.h"
#include "cuda_common.h"
#include "cuda.h"
#include "gpu.h"
#include "gpu_print.h"
#include "print.h"
#include "util.h"
#include "ppcg_options.h"
#include "ca.h"

#define MAX_TAU 0.95
#define MIN_TAU 0.05
#define DFS_STACK_SIZE 250
#define abs(x) ((x < 0 ? -x : x))
#define max(x,y) ((x > y ? x : y))
//#define DEBUG
int graphSize = 0;
int edgeId = 0;
int num_regression_vec = 0;
int unroll_start;
int tile_start;
int texture_start;
int reduction_start;
int skewing_start;
int partition_start;
int inner_tile_start;

int current_iteration = 10;

int best_first_search = 0;
int depth_first_search = 0;
int *dfs_stack;
int dfs_stack_ptr=0;

/* 1 if the last leaf node of the current branch was explored */
int dfs_full_explored=0;
int dfs_full_explore_depth;

int dfs_full_explored_n=0;
int dfs_full_explore_depth_n;
int random_search = 0;
int is_new_sample=0;

int beam_depth;
int beam_edge;
int current_depth;
int semantic_edge;
int n_possible_edges;

int bfs_next_edge;
int current_bfs_next_edge;
int kernel_change=0;
int choose_best = 0;

struct ca_node *beam_node;
struct path *beam_path;
struct path *current_path;

struct regression_sample *samples;

struct node_split {
  int attr;
  int split;
};

struct scc_vec {
  /* Vector of sccs */
  int *sccs;

  /* Length of the vector */
  int len;
};

struct tuple {
  char *key;
  struct ca_node *val;
  
  struct tuple *next;
};

struct kern_time_tuple {
  char *key;
  unsigned long time;
  int push;
  struct ca_ant *ant;

  struct kern_time_tuple *next;
};

struct tuple *node_table;
struct kern_time_tuple *kernel_best_time_table;

struct ca_node *create_node(int type, const char* name, void *val);
int part_size_graph(struct ca_node *node, struct simple_edges *edges, 
		     const int len, int n_part, int n_stmts, int scc);

char **get_schedule_stmt_array(isl_schedule_node *node, int n, int *size);

void build_root(isl_schedule_node *node, struct ca_graph *graphs, 
		struct simple_edges *scc_graph, const int n_part, const int n_stmts);

int **ca_traverse_sched_graph(struct ca_ant *ant, struct coding_ants *ca, struct ca_node *node, 
			     struct path *path_list, int *idx, 
			      int **scc_part, int *start_part, const int lev);

int get_node_kernel(struct ca_ant *ant, struct ca_node *node);
struct ca_ant *copy_ant(struct ca_ant ant);

// log base 2
int lg(int i) {
  int l = 0;
  while ( i >>= 1 ) 
    ++l;

  return l;
}

void push(int e) {
  assert (dfs_stack_ptr < DFS_STACK_SIZE);
  dfs_stack[dfs_stack_ptr++] = e;
}

int pop() {
  return dfs_stack[--dfs_stack_ptr];
}

int peek() {
  return dfs_stack[dfs_stack_ptr - 1];
}

void free_regression_sample(struct regression_sample *set, const int size) {
  int i;
  for (i = 0; i < size; ++i) {
    free(set[i].attr);
  }
  free(set);
}

void copy_regression_sample(struct regression_sample *dst, struct regression_sample *src) {
  dst->attr = (int *)calloc(num_regression_vec, sizeof(int));
  memcpy(dst->attr, src->attr, num_regression_vec * sizeof(int));
  dst->score = src->score;
  memcpy(dst->metric, src->metric, 6 * sizeof(double));
}

char **copy_str_arr(char **arr, const int len) {
  int i;
  char **narr = (char **)calloc(len, sizeof(char *));
  for (i = 0; i < len; ++i) {
    narr[i] = strdup(arr[i]);
  }
  return narr;
}

char *int_to_string(int *vec, const int len) {
  int i;
  if (vec == NULL) {
    return "";
  }

  char *str = (char *)calloc(1000, sizeof(char));
  strcpy(str, "(");
  for (i = 0; i < len; ++i) {    
    char tmp[30];
    snprintf(tmp, 30, "%d, ", vec[i]);
    strcat(str, tmp);
  }

  strcat(str, ")");
  return str;
}

char *float_to_string(float *vec, const int len) {
  int i;
  if (vec == NULL) {
    return "";
  }

  char *str = (char *)calloc(1000, sizeof(char));
  strcpy(str, "(");
  for (i = 0; i < len; ++i) {    
    char tmp[30];
    snprintf(tmp, 30, "%0.4e, ", vec[i]);
    strcat(str, tmp);
  }

  strcat(str, ")");
  return str;
}

char *double_to_string(double *vec, const int len) {
  int i;
  if (vec == NULL) {
    return "";
  }

  char *str = (char *)calloc(1000, sizeof(char));
  strcpy(str, "(");
  for (i = 0; i < len; ++i) {    
    char tmp[30];
    snprintf(tmp, 30, "%0.4e, ", vec[i]);
    strcat(str, tmp);
  }

  strcat(str, ")");
  return str;
}

char *str_arr_to_string(char **vec, const int len) {
  int i;
  printf("str_arr_to_string\n");
  if (vec == NULL)
    return "";
  
  char *str = (char *)calloc(1000, sizeof(char));
  strcpy(str, "(");
  for (i = 0; i < len; ++i) {    
    char tmp[30];
    snprintf(tmp, 30, "%s, ", vec[i]);
    strcat(str, tmp);
  }

  strcat(str, ")");
  return str;

}

int *get_stmt_ids(char **stmts, const int nstmts) {
  int i;
  int *int_stmt = (int *)calloc(nstmts, sizeof(int));
  for (i = 0; i < nstmts; ++i) {
    int_stmt[i] = atoi(&stmts[i][2]);   
  }

  return int_stmt;
}

/* Compute a unique string based on the statements and 
 * the current scheduling row.
 */
char *hash(struct simple_edges *graph, int partition) {
  int i, j;
  char *str = (char *)calloc(1000, sizeof(char));
  if (partition == 1)
    strcpy(str, "P:");
  else if (partition == 0)
    strcpy(str, "S:");
  else if (partition == 2) // reduction
    strcpy(str, "R:");

  // set the row
  //snprintf(str, 30, "(%d)", graph->n_row);
  for (i = 0; i < graph->n_scc; ++i) {
    for (j = 0; j < graph->node[i].n; ++j) {
      char tmp[1000];
      snprintf(tmp, 1000, "%s:", graph->node[i].spaces[j]);
      strcat(str, tmp);
    }
  }
  
  return str;
}

char *hash_kern(struct ca_ant *ant, struct ca_node *node) {
  if (!node->in_kernel)
    return NULL;

  if (node->stmt_str) {
    return str_arr_to_string(node->stmt_str, node->stmt_len);
  }
  
  if (node->kernel_id < 0)
    return NULL;

  if (node->kernel_id >= ant->kernel_stmt_id)
    return NULL;

  printf("using ant\n");
  printf("kernel id = %d, ant stmt id = %d\n", node->kernel_id, ant->kernel_stmt_id);
  printf("len = %d\n", ant->kernel_stmts_len[node->kernel_id]);
  printf("%s\n", ant->kernel_stmts[node->kernel_id][0]);
  
  printf("kern string %s\n", str_arr_to_string(ant->kernel_stmts[node->kernel_id], 
				   ant->kernel_stmts_len[node->kernel_id]));

  return str_arr_to_string(ant->kernel_stmts[node->kernel_id], 
  			   ant->kernel_stmts_len[node->kernel_id]);
}

struct kern_time_tuple *add_kern_tuple(char *key) {
  struct kern_time_tuple *entry = (struct kern_time_tuple *)calloc(1, sizeof(struct kern_time_tuple));
  entry->key = key;
  entry->time = -1;
  entry->next = NULL;  
  
  return entry;
}

/* If key does not exist, add it. Otherwise, 
 * set the time to the min of time and val.
 */
struct kern_time_tuple *kern_table_lookup(char *key) {
  if (kernel_best_time_table == NULL) {
    kernel_best_time_table = add_kern_tuple(key);
    return kernel_best_time_table;
  }

  struct kern_time_tuple *trav = kernel_best_time_table;
  struct kern_time_tuple *ptrav = NULL;
  while (trav != NULL) {
    if (strcmp(trav->key, key) == 0) {
      return trav;
    }
    ptrav = trav;
    trav = trav->next;
  }
  assert(ptrav != NULL);
  ptrav->next = add_kern_tuple(key);
  return ptrav->next;
}

void set_min_kern(struct ca_ant *ant, struct ca_node *node) {
  printf("set_min_kern");
  char *key = hash_kern(ant, node);
  if (key == NULL)
    return;

  struct kern_time_tuple *tup = kern_table_lookup(key);
  assert(tup != NULL);

  if (tup->time > node->kernel_score || tup->time <= 0) {
    tup->time = node->kernel_score;
    tup->ant = copy_ant(*ant);
    tup->push = 0;
    kernel_change = 1;
  }
}

// add entry into the node table
struct tuple* add_entry(char *key) {
  printf("adding entry %s\n", key);
  struct tuple *entry = (struct tuple *)calloc(1, sizeof(struct tuple));
  entry->key = key;
  entry->val = create_node(SOURCE, key, NULL);
  entry->next = NULL;
  return entry;
}


struct ca_node *node_lookup_by_key(char *key) {
  if (node_table == NULL) {
    node_table = add_entry(key);
    assert(node_table != NULL);
    return node_table->val;
  }

  struct tuple *trav = node_table;
  struct tuple *ptrav = NULL;
  while (trav != NULL) {
    //printf("\t %s == %s? \n", trav->key, key);
    if (strcmp(trav->key, key) == 0) {
      // match
      printf("match %s, %s\n", trav->key, key);
      return trav->val;
    }
    
    ptrav = trav;
    trav = trav->next;
  }
  
  assert (ptrav != NULL);
  //ptrav->next = (struct tuple *)calloc(1, sizeof(struct tuple));
  //ptrav->key = key;
  //ptrav->val = create_node(SOURCE, key, NULL);
  // add new entry
  ptrav->next = add_entry(key);
  return ptrav->next->val;
}

/* Get the node based on the statements and n_row in 
 * 'graph'. If it does not exist, then create a new 
 * node and put it in the node_table
 */
struct ca_node *node_lookup(struct simple_edges *graph, int partition) {
  char *key = hash(graph, partition);
  
  if (node_table == NULL) {
    node_table = add_entry(key);
    assert(node_table != NULL);
    return node_table->val;
  }

  struct tuple *trav = node_table;
  struct tuple *ptrav = NULL;
  while (trav != NULL) {
    //printf("\t %s == %s? \n", trav->key, key);
    if (strcmp(trav->key, key) == 0) {
      // match

      return trav->val;
    }
    
    ptrav = trav;
    trav = trav->next;
  }
  
  assert (ptrav != NULL);
  //ptrav->next = (struct tuple *)calloc(1, sizeof(struct tuple));
  //ptrav->key = key;
  //ptrav->val = create_node(SOURCE, key, NULL);
  // add new entry
  ptrav->next = add_entry(key);
  return ptrav->next->val;
}

// return a copy of arr
int *int_copy(int *arr, int len) {
  if (arr == NULL) {
    return NULL;
  }

  int *cpy = (int *)calloc(len, sizeof(int));
  memcpy(cpy, arr, len * sizeof(int));
  return cpy;
}

// print graph in a convenient ascii format
void print_graph(struct ca_node *node, char *tabs, FILE *fp) {
  int i;
  int start_flag = 0;

  if (tabs == NULL) {
    start_flag = 1;
    tabs = (char *)calloc(1000, sizeof(char));
    strcpy(tabs, "");
  }
  
  if (node == NULL || node->out_edges == NULL || node->visited) {
    return;
  }

  const int n_edges = node->n_edges;
  fprintf(fp, "%s\n", node->name);

  node->visited = 1;

  char buf[1000];
  fprintf(fp, "%s |\n", tabs);
  for(i = 0; i < n_edges; ++i) { 
    strcpy(buf, tabs); 
    fprintf(fp, "%s %0.4e-->", buf, node->out_edges[i].tau);
    strcat(buf, "\t\t");
    
    if (node->out_edges[i].dst == NULL) {
      fprintf(fp, "EMPTY\n");
      continue;
    }

    print_graph(node->out_edges[i].dst, buf, fp);
    if (i != n_edges-1) {
      fprintf(fp, "%s |\n", tabs);
    }
  }
  
  node->visited = 0;
  if (start_flag) {
    // we are done, free tabs
    free(tabs);
  }
}

void print_graph_to_file(const char* filename, struct ca_node *source) {
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("ERROR opening file\n");
    exit(1);
  }

  print_graph(source, NULL, fp);

  fclose(fp);
}


// return true if string is in array
int exists(char *string, char **array, int n_array) {
  int i;
  for (i = 0; i < n_array; ++i) {
    if (strcmp(array[i], string) == 0) {
      return 1;
    }
  }

  return 0;
}

struct path *get_tail(struct path *path_list) {
  struct path *trav = path_list->next->next;
  while(trav != NULL) {
    trav = trav->next;
    path_list = path_list->next;
  }	
  assert (path_list->next != NULL);
  assert (path_list->next->next == NULL);
  
  return path_list;
}

struct coding_ants *coding_ants_alloc(const char* input) {
  struct coding_ants *ca = (struct coding_ants*)calloc(1, sizeof(struct coding_ants));
  ca->input = strdup(input);  
  ca->config.tex_array = NULL;
  ca->config.n_tex_array = 0;
  ca->init = 1;
  return ca;
}

void print_val(struct ca_node node) {
  printf("%s = ", node.name);
  if (node.type == COMPILER_OPTION) {
    printf("%d\n", *((int *)node.val));
  } 
  else if (node.type == MEM_PLACE) {
    printf("%d\n", *((int *)node.val));
  } 
  else {
    printf("%s\n", ((char *)node.val));
  }
}

/*
 * Compute the integer power of x^y.
 */
int int_pow(const int x, const int y){
  assert(y >= 0);

  int i;
  int val = 1; 
  
  for (i = 0; i < y; ++i) {
    val *= x;
  }

  return val;
}

char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

// split 'arr' based on ',' and convert the array of strings to 
// an integer array
int *get_parallel_dims_as_int(char *arr, const int block_dim) {
  assert(block_dim > 0);
  int i;
  int *int_dims = (int *)calloc(block_dim, sizeof(int));

  char buf[16];
  strcpy(buf, arr);

  char **split_dims = str_split(buf, ',');
  for (i = 0; i < block_dim; ++i) {
    int_dims[i] = atoi(split_dims[i]);
  }
  
  return int_dims;
}

// true if every value in 'v1' is less than or equalt to the
// corresponding element in 'v2'
int vec1_lt_vec2(int *v1, int *v2, int dim) {
  int i;
  for (i = 0; i < dim; ++i) {
    if (v1[i] > v2[i]) {
      return 0;
    }
  }
  
  return 1;
}

// true if every value in 'v1' is equal to the
// corresponding element in 'v2'
int vec1_eq_vec2(int *v1, int *v2, int dim) {
  int i;
  for (i = 0; i < dim; ++i) {
    if (v1[i] != v2[i]) {
      return 0;
    }
  }
  
  return 1;
}

int int_exists(const int v, int *vec, const int n) {
  int i;
  for (i = 0; i < n; ++i) {
    if (vec[i] == v) {
      return 1;
    }
  }

  return 0;
}

void print_scc_graph(struct simple_edges *graph) {
  int i;
  for (i = 0; i < graph->n_scc; ++i) {
    printf("scc %d, n_unsat = %d\n", graph->node[i].scc, graph->node[i].n_unsat);
  }
}

// reset the n_unsat to n_in
void reset_graph(struct simple_edges *graph) {
  int i;
  for (i = 0; i < graph->n_scc; ++i) {
    graph->node[i].n_unsat = graph->node[i].n_in;
  }
}

//get the number of sccs in mask, with 0 unsatisifed predecessors 
int get_num_valid_sccs(struct simple_edges *graph, int *mask, const int len) {
  int i;
  int n_valid = 0;
  for (i = 0; i < len; ++i) {
    if (graph->node[mask[i]].n_unsat == 0) {
      ++n_valid;
    }
  }

  return n_valid;
}

int get_num_valid_sccs2(struct simple_edges *graph) {
  int i;
  int n_valid = 0;
  for (i = 0; i < graph->n_scc; ++i) {
    if (graph->node[i].n_unsat == 0) {
      ++n_valid;
    }
  }

  return n_valid;
}

int *get_valid_sccs2(struct simple_edges *graph) {
  int i;

  int idx = 0;

  int n_valid = get_num_valid_sccs2(graph);
  int *valid = (int *)calloc(n_valid, sizeof(int));
  for (i = 0; i < graph->n_scc; ++i) {
    if (graph->node[i].n_unsat == 0) {
      valid[idx++] = graph->node[i].scc;
    }
  }
  
  return valid;
}

//get the sccs with 0 unsatisifed predecessors
int *get_valid_sccs(struct simple_edges *graph, int *n_valid, int *mask, const int len) {
  assert (len <= graph->n_scc);

  int i;

  *n_valid = get_num_valid_sccs(graph, mask, len);
  
  int idx = 0;
  int *valid = (int *)calloc(*n_valid, sizeof(int));
  for (i = 0; i < len; ++i) {
    if (graph->node[mask[i]].n_unsat == 0) {
      valid[idx++] = graph->node[mask[i]].scc;
    } 
  }
  
  return valid;
}

int get_num_sccs(struct simple_edges *graph, int *stmts, const int n) {
  int i, j;
  
  int scc_ct = 0;
  // for each scc
  for (i = 0; i < graph->n_scc; ++i) {
    // for each stmt
    for (j = 0; j < n; ++j) {
      // does the stmt id exist in the scc 
      if (int_exists(stmts[j], graph->node[i].stmts, graph->node[i].n)) {
	++scc_ct;
	break;
      }
    }
  }
  
  return scc_ct;
}

// get a list of sccs which 'stmts' belong
int *get_sccs(struct simple_edges *graph, int *stmts, const int n, int *n_scc) {
  int i, j;
  *n_scc = get_num_sccs(graph, stmts, n);
  assert (*n_scc > 0);

  int *sccs = (int *)calloc(*n_scc, sizeof(int));

  int scc_idx = 0;
  // for each scc node
  for (i = 0; i < graph->n_scc; ++i) {
    for (j = 0; j < n; ++j) {
      if (int_exists(stmts[j], graph->node[i].stmts, graph->node[i].n)) {
	sccs[scc_idx++] = graph->node[i].scc;
	break;
      }
    }
  }

  return sccs;
}

int *get_sccs_from_stmts(struct simple_edges *graph, 
			 const int n_stmts, isl_schedule_node *node, int *n_scc) {
  int i, j;
  int n;
  int len = 0;

  n = n_stmts;

  char **domain_stmt_arr = graph->domain_stmt_array;//get_stmt_array(dom);
  char **sched_stmt_arr = get_schedule_stmt_array(node, n, &len);
  for (i = 0; i < n; ++i) {
    printf("%s, ", domain_stmt_arr[i]);
  }
  printf("\n");
  
  // get the statement ids of the stmts in under node
  int *node_stmts = (int *)calloc(len, sizeof(int));
  int idx = 0;
  for (i = 0; i < len; ++i) {
    for (j = 0; j < n; ++j) {
      if (strcmp(sched_stmt_arr[i], domain_stmt_arr[j]) == 0) {
	printf("stmt : %s: %d\n", sched_stmt_arr[i], j);
	node_stmts[idx++] = j;
      }
    }
  }

  free(sched_stmt_arr);
  return get_sccs(graph, node_stmts, len, n_scc);  
}

void decay_edge(struct ca_edge *edge) {
  edge->tau = (1 - RHO) * edge->tau;
}

struct ca_node *create_node(int type, const char* name, void *val) {
  ++graphSize;
  struct ca_node *node;
  node = (struct ca_node*)calloc(1, sizeof(struct ca_node));
  node->type = type;
  
  node->name = strdup(name);//(char *)calloc(50, sizeof(char));
  //strcpy(node->name, name);

  node->val = val;
  node->out_edges = NULL;
  node->aux = NULL;
  node->subgraph = NULL;
  node->n_subgraph = 0;
  node->visited = 0;
  node->kernel_id = -1;
  node->stmt_str = NULL;
  node->stmt_len = 0;
  node->in_kernel = 0;
  node->best_score = 1e9;
  node->kernel_score = -1;
  return node;
}

void init_edge_weight(struct ca_edge *edge, const int n_edges){
  int i;
  edge->has_been_traversed = 0;
  assert (n_edges > 0);

  edge->tau = (1.0 / (double) n_edges);
  edge->tau_start = edge->tau;
  edge->metrics = (float **)calloc(2 * NUM_ANTS * NUM_ITER, sizeof(float *));;
  for (i = 0; i < 2 * NUM_ANTS * NUM_ITER; ++i) {
    edge->metrics[i] = (float *)calloc(6, sizeof(float));
  }
  edge->metric_ptr = 0;
  edge->best_metrics_id = -1;
  edge->semantic_edge = 0;
}

void ca_allocate_implicit_edge(struct ca_node *node, const int n_edges) {
  node->implicit_edge = (struct ca_edge*)calloc(1, sizeof(struct ca_edge));
  node->implicit_edge->dst = NULL;
  node->implicit_edge->src = node;
  node->n_implicit = n_edges;
  
  init_edge_weight(node->implicit_edge, n_edges);
}

void ca_realloc_edges(struct ca_node *node, const int n_edges) {
  printf("realloc edges\n");
  int i;
  if (n_edges <= node->n_edges) {
    return;
  }

  int o_edges = node->n_edges;

  assert(o_edges < n_edges);
  
  node->out_edges = (struct ca_edge*)realloc(node->out_edges, n_edges * sizeof(struct ca_edge));
  assert (node->out_edges);
  
  node->n_edges = n_edges;
 
  // allocate space for edge weights and set default value
  //node->edge_weights = (double *)calloc(n_children, sizeof(double));
  for (i = o_edges; i < n_edges; ++i) {
    init_edge_weight(&node->out_edges[i], n_edges);
    node->out_edges[i].edge_id = edgeId++;
    //node->out_edges[i].tau = (1.0 / (double) n_edges);
    //node->out_edges[i].tau_start = node->out_edges[i].tau;
    //node->out_edges[i].eta = 1.0;
    node->out_edges[i].src = node;
    node->out_edges[i].dst = NULL;
  }
  printf("end realloc edges\n");
}

void ca_allocate_out_edges(struct ca_node *node, const int n_edges) {
  int i;
  if (n_edges == 0) {
    node->out_edges = NULL;
    node->n_edges = n_edges;
    return;
  }

  assert (n_edges > 0);
  
  node->out_edges = (struct ca_edge*)calloc(n_edges, sizeof(struct ca_edge));
  node->n_edges = n_edges;

  // allocate space for edge weights and set default value
  //node->edge_weights = (double *)calloc(n_children, sizeof(double));
  for (i = 0; i < n_edges; ++i) {
    init_edge_weight(&node->out_edges[i], n_edges);
    node->out_edges[i].edge_id = edgeId++;
    //node->out_edges[i].tau = (1.0 / (double) n_edges);
    //node->out_edges[i].tau_start = node->out_edges[i].tau;
    //node->out_edges[i].eta = 1.0;
    node->out_edges[i].src = node;
    node->out_edges[i].dst = NULL;    
  }
}

// return the number options based on the block dimension
int num_parallel_opts(int block_dim) {
  assert(block_dim > 0 && block_dim <= 3);
  return (block_dim == 1 ? 6 : (block_dim == 2 ? 39 : 156));
}

char **parallel_dim_opts_1d() {
  char *xvals[] = {"32", "64", "128", "256", "512", "1024"};

  int i;
  const int n_opts = num_parallel_opts(1);
  char **arr = (char **)calloc(n_opts, sizeof(char*));
  
  for (i = 0; i < n_opts; ++i) {    
    arr[i] = (char *)calloc(5, sizeof(char));
    strcpy(arr[i], xvals[i]);
  }
  
  return arr;
}

char **parallel_dim_opts_2d() {
  int i, j;
  
  const int n_opts = num_parallel_opts(2);
  char **arr = (char **)calloc(n_opts, sizeof(char*));
  
  char *xvals[] = {"1", "8", "16", "32", "64", "128", "256", "512", "1024"};
  char *yvals[] = {"1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"};

  int pos = 0;
  for (i = 0; i < 9; ++i) {
    for (j = 0; j < 11; ++j) { 
      int x = atoi(xvals[i]);
      int y = atoi(yvals[j]);
      if (x * y > 1024 || x * y < 32) {
	continue;
      }

      arr[pos] = (char *)calloc(50, sizeof(char));
      strcpy(arr[pos], xvals[i]);
      strcat(arr[pos], ",");
      strcat(arr[pos], yvals[j]);
      ++pos;
    }
  }

  return arr;
}

char **parallel_dim_opts_3d() {
  int i, j, k;
  
  const int n_opts = num_parallel_opts(3);
  char **arr = (char **)calloc(n_opts, sizeof(char*));
  
  char *xvals[] = {"1", "8", "16", "32", "64", "128", "256", "512", "1024"};
  char *yvals[] = {"1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"};
  char *zvals[] = {"1", "2", "4", "8", "16", "32", "64"};

  int pos = 0;
  for (i = 0; i < 9; ++i) {
    for (j = 0; j < 11; ++j) {
      for (k = 0; k < 7; ++k) {       
	int x = atoi(xvals[i]);
	int y = atoi(yvals[j]);
	int z = atoi(zvals[k]);
	
	if (x * y * z > 1024 || x * y * z < 32) {
	  continue;
	}
	arr[pos] = (char *)calloc(50, sizeof(char));
	strcpy(arr[pos], xvals[i]);
	strcat(arr[pos], ",");
	strcat(arr[pos], yvals[j]);
	strcat(arr[pos], ",");
	strcat(arr[pos], zvals[k]);	
	++pos;
      }
    }
  }

  return arr;
}

char **tile_dim_opts() {
  int i;

  // collect different tile factors from 4 to 256
  char **arr = (char **)calloc(8, sizeof(char*));
  
  int dim_val = 1;
  for (i = 0; i < 8; ++i) {    
    arr[i] = (char *)calloc(4, sizeof(char));

    snprintf(arr[i], 4, "%d", dim_val);
    if (dim_val == 1) {
      dim_val = 4;
    } else {
      dim_val *= 2;
    }
  }
  
  return arr;
}

void populate_options(char **arr, int *pos, char **parallel_dims, char **seq_dims, 
		      int cdim, int total_dim, int block_dim, char *val) {
  int i;
  if (cdim >= total_dim) {
    // base case record the current string
    arr[(*pos)] = (char *)calloc(100, sizeof(char));
    strcpy(arr[(*pos)++], val); 
    return;
  }

  int par_dim_len = num_parallel_opts(block_dim);
  int seq_dim_len = 8;

  int ub = (cdim < block_dim ? par_dim_len : seq_dim_len);
  for (i = 0; i < ub; ++i) {
    char cval[100];
    strcpy(cval, val);
    if (strcmp(val, "") != 0) {
      strcat(cval, ",");
    }

    if (cdim < block_dim) { // parallel dimension
      strcat(cval, parallel_dims[i]);
      populate_options(arr, pos, parallel_dims, seq_dims, cdim + block_dim, 
		       total_dim, block_dim, cval);
    } 
    else { // sequential dimension
      strcat(cval, seq_dims[i]);
      populate_options(arr, pos, parallel_dims, seq_dims, cdim + 1, 
		       total_dim, block_dim, cval);
    }

  }
 
} 

/* 
 * Get an array of string corresponding to different
 * dimensions based on dim.
 */
char **tile_dim_str(const int dim, const int block_dim) {
  assert(dim >= block_dim);

  int i; // loop iterators

  // an array parallel tile size options
  char **parallel_dim_opts;

  // an array of sequential tile size options
  char **seq_dim_opts;
  int par_dim_len = num_parallel_opts(block_dim);
  switch(block_dim) {
  case 1:
    parallel_dim_opts = parallel_dim_opts_1d();
    break;
  case 2:
    parallel_dim_opts = parallel_dim_opts_2d();
    break;
  case 3:
    parallel_dim_opts = parallel_dim_opts_3d();
    break;
  default:
    printf("ERROR: unexpected block dim\n");
    exit(1);
  }
  
  seq_dim_opts = tile_dim_opts(); 
  const int n_opts = par_dim_len * int_pow(8, (dim - block_dim));

  char **tile_opts = (char **)calloc(n_opts, sizeof(char *));
  int pos = 0;
  populate_options(tile_opts, &pos, parallel_dim_opts, seq_dim_opts, 
		   0, dim, block_dim, "");

  return tile_opts;
}

void test() {
  int i, j;

  int dim = 1;
  int block_dim = 1;
  int par_dim_len = num_parallel_opts(block_dim);

  const int len = par_dim_len * int_pow(8, (dim - block_dim));
  char **arr = tile_dim_str(dim, block_dim);

  for (i = 0; i < len; ++i) {
    printf("%s\n", arr[i]);
    free(arr[i]);
  }
  free(arr);
}

void set_cuda_dims(struct ca_node *source, const int kernel_id,
		   const int tile_dim, const int block_dim) {
  int i, j; // loop iterators
  const int par_dim_len = num_parallel_opts(block_dim);
  
  // the number of tile options
  const int len = par_dim_len * int_pow(8, (tile_dim - block_dim));

  // (3) get the tile options 
  char **arr = tile_dim_str(tile_dim, block_dim);

  // each tile option maps to a child node in the graph
  ca_allocate_out_edges(source, len);

  // (4) create a node for each tile option
  for (i = 0; i < len; ++i) {
    char buf[50];
    snprintf(buf, 50, "tile_size:%d", i);
    
    char *val = (char *)calloc(100, sizeof(char));
    snprintf(val, 100, "kernel[%d]->tile[%s]", kernel_id, arr[i]);

    struct ca_node *node = create_node(CUDA_CONFIG, buf, val);    
    
    char **par_arr = (block_dim == 1 ? parallel_dim_opts_1d() : 
		      (block_dim == 2 ? parallel_dim_opts_2d() : 
		       parallel_dim_opts_3d()));

    // (5) create a node for each block size option

    // get the upper bound on the thread block dim. as defined by
    // the current tile size
    int *max_block_dims = get_parallel_dims_as_int(arr[i], block_dim);
      
    // figure out how many children this node will have
    int num_children = 0;
    for (j = 0; j < par_dim_len; ++j) {
      int *int_block_dims = get_parallel_dims_as_int(par_arr[j], block_dim);

      // block dims > max dims does not impact the generated
      // code. We skip it in this case	
      if (vec1_lt_vec2(int_block_dims, max_block_dims, block_dim)) {
	++num_children;
      }
	
      // cleanup
      free(int_block_dims);
    }

    // allocate num_children
    ca_allocate_out_edges(node, num_children);

    int child_idx = 0;
    for (j = 0; j < par_dim_len; ++j) {
      int *int_block_dims = get_parallel_dims_as_int(par_arr[j], block_dim);

      // block dims > max dims does not impact the generated
      // code. We skip it in this case	
      if (!vec1_lt_vec2(int_block_dims, max_block_dims, block_dim)) {
	free(par_arr[j]);
	continue;
      }

      char blk_buf[50];
      snprintf(blk_buf, 50, "block_size:%d", child_idx);

      char *blk_val = (char *)calloc(100, sizeof(char));
      snprintf(blk_val, 100, "kernel[%d]->block[%s]", kernel_id, par_arr[j]);

      struct ca_node *block_node = create_node(CUDA_CONFIG, blk_buf, blk_val);
      node->out_edges[child_idx++].dst = block_node;

      // cleanup 
      free(par_arr[j]);
      free(int_block_dims);
    } // end for j
      
    source->out_edges[i].dst = node;
      
    // cleanup
    free(par_arr);
    free(max_block_dims);
    free(arr[i]); 
    
  } // end for i
    

  // cleanup
  free(arr);
}


// return an array of read_only arrays and set n_array to the number of read_only arrays
struct ca_array *get_read_only_arrays(struct coding_ants *ca, int *n_ro_array) {
  int i;
  struct ca_array *ro_arrays;

  *n_ro_array = 0;

  for (i = 0; i < ca->n_array; ++i) {
    if (ca->arrays[i].read_only) {
      ++(*n_ro_array);
    }
  }
  
  ro_arrays = (struct ca_array *)calloc(*n_ro_array, sizeof(struct ca_array));
  
  int ro_ptr = 0;
  for (i = 0; i < ca->n_array; ++i) {
    if (ca->arrays[i].read_only) {
      ro_arrays[ro_ptr++] = ca->arrays[i];
    }
  }

  return ro_arrays;
}

// Build the mem opt graph in a dfs manner. Each level of the graph contains two options for
// a specific array (texture or global)
void dfs_build_mem_graph(struct ca_node *node, struct ca_array *arrays, int n_array, int idx) {
  int i;
  if (idx == n_array) {
    // base case
    return;
  }

  // only consider texture and global memory for now
  const int n_mem_place = 2; 

  ca_allocate_out_edges(node, n_mem_place);
  for (i = 0; i < n_mem_place; ++i) {
    char node_name[50];
    snprintf(node_name, 50, "%s", arrays[idx].name);
    
    // flag 0 -> global mem, 1 -> texture
    int *v = (int *)calloc(1, sizeof(int));
    *v = i;

    struct ca_node *child = create_node(MEM_PLACE, node_name, v);
    
    // dfs recurse
    dfs_build_mem_graph(child, arrays, n_array, idx + 1);
    
    node->out_edges[i].dst = child;    
  }
}

// set the graph for memory placement
void set_mem_opt_graph(struct ca_node *source, struct coding_ants *ca) {
  int i;
  
  int n_array;
  struct ca_array *ro_arrays = get_read_only_arrays(ca, &n_array);
  dfs_build_mem_graph(source, ro_arrays, n_array, 0);
}

// add the compiler options for shared memory usage
void set_shared_mem_graph(struct ca_node *source) {
  int i;
    
  ca_allocate_out_edges(source, 2);

  // set shared memory nodes
  for (i = 0; i < 2; ++i) {
    char buf[50];
    snprintf(buf, 50, "shared_mem:%d", i);
    
    int *v = (int *)calloc(1, sizeof(int));
    *v = i;

    struct ca_node *shared = create_node(COMPILER_OPTION, buf, v);    
    source->out_edges[i].dst = shared;    
  }

}

char *get_program_name(struct coding_ants *ca) {
   // last forward slash in input
  char *fs = strrchr(ca->input, '/');
  
  // last '.' in input
  char *dot = strrchr(ca->input, '.');

  char *name = (char *)calloc(50, sizeof(char));
  memcpy(name, (fs + 1), (dot - fs - 1));

  return name;
}

pid_t child_pid = -1;
char global_program_name[100];
void kill_child(int sig) {
  kill(child_pid, SIGINT);  
  char skill[100];
  
  // the name of the program in 'top' is only the 
  // first 15 characters
  char *topname = (char *)calloc(16, sizeof(char));
  strncpy(topname, global_program_name, 15);
  strcpy(skill, "skill ");
  strcat(skill, topname);//global_program_name);
  
  printf("killing child with %s\n", skill);
  if (system(skill) < 0) {
    printf("system error\n");
    exit(1);
  }

  if (system("./gpu_reset/resetGPU") < 0) {
    printf("system error\n");
    exit(1);
  }
  alarm(0);
}

// run either the cuda or original code depending on 'type'
double get_run_time(struct coding_ants *ca, const char* type) {
  assert(strcmp(type, "ppcg_cuda") == 0 || strcmp(type, "orig") == 0);

  FILE *fp, *out;
  char path[1035];
  double time = 1000000000.0;
  char *name = get_program_name(ca);
  
  strcpy(global_program_name, name);
  strcat(global_program_name, ".");
  strcat(global_program_name, type);

  char prog_name[100];
  strcpy(prog_name, "./out.ppcg-0.05/");
  strcat(prog_name, name);
  strcat(prog_name, ".");
  strcat(prog_name, type);//ppcg_cuda"); 

   
  char command[100];
  strcpy(command, "");
  
  strcat(command, prog_name);
  printf("running=%s\n", command);
  
  signal(SIGALRM,(void (*)(int))kill_child);
  
  int pipefd[2];
  if (pipe(pipefd) < 0) {
    printf("pipe error\n");
    exit(0);
  }
  
  char buffer[4096];
  
  child_pid = fork();
  if (child_pid == 0) { // child
    printf("child\n");

    if (close(pipefd[0]) < 0) {   // close reading end in the child
      printf("child close pipe error\n");
      exit(1);
    }

    if (dup2(pipefd[1], 1) < 0) {  // send stdout to the pipe
      printf("dup2 error\n");
      exit(1);
    }
    
    if (dup2(pipefd[1], 2) < 0) {  // send stderr to the pipe
      printf("dup2 error\n");
      exit(1);
    }

    if (close(pipefd[1]) < 0) {    // this descriptor is no longer needed
      printf("child close pipe error\n");
      exit(0);
    }

    if (execvp(command, NULL) < 0) {
      printf("command error\n");
      exit(1);
    }
    exit(0);
  }
  else {
    if (close(pipefd[1]) < 0) {  // close the write end of the pipe in the parent
      printf("parent close pipe error\n");
      exit(1);
    }

    if(strcmp(type, "ppcg_cuda") == 0) {
      alarm(10);
    }
    else {
      alarm(0);
    }

    char out_name[50];
    strcpy(out_name, prog_name);
    strcat(out_name, ".out");

    out = fopen(out_name, "w");
    if (out == NULL) {
      printf("Failed to open the out file\n");
      exit(1);
    }
    int timing_flag = 1;

    size_t nbytes = 0;
    while ((nbytes = read(pipefd[0], buffer, sizeof(buffer))) != 0)
    {
      if (timing_flag) {
	char *nl = strchr(buffer, '\n');
	char stime[100];
	memset(stime, 0, 100 * sizeof(char));
	memcpy(stime, buffer, (nl - buffer));
	time = atof(stime);
	timing_flag = 0;
	fwrite(nl, sizeof(char), (nbytes - (nl - buffer)), out);
      }
      else {
	fwrite(buffer, sizeof(char), nbytes, out);
      }
      memset(buffer, 0, 4096 * sizeof(char));
    }

    wait(NULL);

    // cancel the alarm
    alarm(0);

    if (close(pipefd[0]) < 0) {  // close the read end of the pipe in the parent
      printf("parent close pipe error\n");
      exit(1);
    }

    fclose(out);
  }
 
  /* Open the command for reading. */
  /*fp = popen(command, "r");
  if (fp == NULL) {
    printf("Failed to run command\n" );
    exit(1);
  }

  char out_name[50];
  strcpy(out_name, prog_name);
  strcat(out_name, ".out");

  out = fopen(out_name, "w");
  if (out == NULL) {
    printf("Failed to open the out file\n");
    exit(1);
  }

  // the first printed is the timing
  int timing_flag = 1;

  /* Read the output a line at a time - output it. */
  /*while (fgets(path, sizeof(path)-1, fp) != NULL) {
    if (timing_flag) {
      time = atof(path);
      timing_flag = 0;
    }
    else {
      fputs(path, out);
    }
  }

  /* close */
  /*pclose(fp);
  fclose(out);
  */
  // cleanup
  free(name);
  if (time == 0) {
    // something went wrong
    return 1e6;
  }
  return time;
}


// compile either the cuda or original code depending on 'type'
void compile(struct coding_ants *ca, const char* type) {
  assert(strcmp(type, "orig") == 0 || strcmp(type, "cuda") == 0);

  FILE *fp;
  char path[1035];
  char command[256];
  
  char bsh[50];
  strcpy(bsh, "./compile_");
  strcat(bsh, type);
  strcat(bsh, ".sh ");
  
  strcpy(command, bsh);//"./compile_cuda.sh ");
  strcat(command, ca->input);
  
  printf("command=%s\n", command);
  
  /* Open the command for reading. */
  fp = popen(command, "r");
  if (fp == NULL) {
    printf("Failed to run command\n" );
    exit(1);
  }

  /* Read the output a line at a time - output it. */
  while (fgets(path, sizeof(path)-1, fp) != NULL) {
    printf("%s", path);
  }

  /* close */
  pclose(fp);
}

// return 0 if the cuda results do not match the original results
int compare_results(struct coding_ants *ca) {
  FILE *fp;
  char path[1024];

  char *name = get_program_name(ca);

  char cuda_prog_name[100];
  char orig_prog_name[100];
  strcpy(cuda_prog_name, "./out.ppcg-0.05/");
  strcat(cuda_prog_name, name);
  strcat(cuda_prog_name, ".ppcg_cuda.out");

  strcpy(orig_prog_name, "./out.ppcg-0.05/");
  strcat(orig_prog_name, name);
  strcat(orig_prog_name, ".orig.out");

  char command[100];
  strcpy(command, "python compare.py -i ");
  strcat(command, orig_prog_name);
  strcat(command, " -o ");
  strcat(command, cuda_prog_name);

  /* Open the command for reading. */
  fp = popen(command, "r");
  if (fp == NULL) {
    printf("Failed to run command\n" );
    exit(1);
  }

   /* Read the output a line at a time - output it. */
  while (fgets(path, sizeof(path)-1, fp) != NULL) {;
    // should not print anything on success
    return 0;
  }
  fclose(fp);

  return 1;
}

void get_metrics(struct coding_ants *ca) {
  FILE *fp;
  char command[1000];
  //nvprof --timeout 15 --csv --log-file tst.log --metrics ipc,achieved_occupancy,gst_throughput,gld_throughput,l2_l1_read_hit_rate,tex_cache_hit_rate ./out.ppcg-0.05/mvt.ppcg_cuda &> ./out.ppcg-0.05/mvt.ppcg_cuda.out

  char *prog_name = get_program_name(ca);
  strcpy(command, "nvprof --timeout 25 --csv --log-file metrics.csv --metrics ipc,achieved_occupancy,gst_throughput,gld_throughput,l2_l1_read_hit_rate,tex_cache_hit_rate --events elapsed_cycles_sm ./out.ppcg-0.05/");
  
  strcat(command, prog_name);
  strcat(command, ".ppcg_cuda &> ./out.ppcg-0.05/");
  strcat(command, prog_name);
  strcat(command, ".ppcg_cuda.out");

  printf("%s\n", command);
  fp = popen(command, "r");
  if (fp == NULL) {
    printf("Failed to run command\n" );
    exit(1);
  }
  
  char buffer[1028];
  
  while (fgets(buffer, 1028, fp) != NULL)
  {
    
    //if(0) 
    //printf("%c\n", buffer[0]);
  } 

  pclose(fp);
}

double evaluate(struct coding_ants *ca, int run_metrics) {
  int i;
  compile(ca, "cuda");
  double cuda_run_time = get_run_time(ca, "ppcg_cuda");

  if (cuda_run_time < 1) {
    // run 2 more times and take the average
    for (i = 0; i < 9; ++i) 
      cuda_run_time += get_run_time(ca, "ppcg_cuda");
    //cuda_run_time += get_run_time(ca, "ppcg_cuda");
    
    cuda_run_time /= 10.0;    
  }

  printf("cuda time = %0.4e, ppcg time = %0.4e, current best = %0.4e, orig time = %0.4e\n", 
	 cuda_run_time, ca->ppcg_only_time, ca->best_current_time, ca->orig_run_time);

  if (cuda_run_time < 1e6 && !compare_results(ca)) {
    printf("ERROR: Incorrect CUDA code...\n");
    exit(1);
    return 1e9;
  }
  
  if (cuda_run_time < 1e6 && run_metrics) {
    get_metrics(ca);    
  }

  if (cuda_run_time == 0)
    return 1e9;

  return cuda_run_time;
}

// perform some initialization to prepare for the application of ant's 
// optimizations
void pre_ca_apply(struct ca_ant *ant, struct coding_ants *ca) {
  printf("pre_ca_apply\n");
  assert (ant != NULL);
  assert(ca->config.tex_array == NULL);
  int i, j;

  ant->n_est = 0;
  memset(ant->est_metrics, 0, 6 * sizeof(float));
  memset(ant->opt_vec, 0, num_regression_vec * sizeof(int));
  
  if (ant->kernel_stmts) {
    for (i = 0; i < ant->kernel_stmt_id; ++i) {
      for (j = 0; j < ant->kernel_stmts_len[i]; ++j) {
	free(ant->kernel_stmts[i][j]);
      }
      free(ant->kernel_stmts[i]);
    }
    free(ant->kernel_stmts_len);
    free(ant->kernel_stmts);
    
    ant->kernel_stmts = NULL;
    ant->kernel_stmt_id = 0;
  }

  // default to no split
  for (i = 0; i < num_regression_vec; ++i) {
    //ant->filter.attr_vec[i] = -1;
    ant->filter.attr_lb[i] = -1;
    ant->filter.attr_ub[i] = -1;
  }

  //memset(ant->filter.lte_mask, 0, num_regression_vec * sizeof(int));
  ant->filter_set = 0;
  ant->opt_vec_ptr = 0;
  ant->do_reduction = 0;
  ant->first_partition = 1;
  // get number of textures
  int n_tex_array = 0;
  for (i = 0; i < ca->mem_opt_graph.depth; ++i) {
    if(*((int *)ant->mem_opts[i].val)) {
      // texture memory is used
      ++n_tex_array;
    }
  }
  // set the texture array
  ca->config.n_tex_array = n_tex_array;
  ca->config.tex_array = (char **)calloc(n_tex_array, sizeof(char *));

  //assert (ca->edges->max_dim > 0);
  //ca->config.sccs = (int **)calloc(ca->edges->max_dim, sizeof(int *));
  //for (i = 0; i < ca->edges->max_dim; ++i) {
    //ca->config.sccs[i] = NULL;
  //}
  printf("end pre_ca_apply\n");
}

void set_ca_leaf_aux(isl_schedule_node *node_sc, struct ca_node *node_ca, 
		     struct path *path_list, int idx, const int band_num) {
  int i;

  if (node_ca->out_edges == NULL) {
    assert(node_ca->type == PART_SIZE);

    // no subgraphs
    if (node_ca->n_subgraph == 0) {
      node_ca->aux = isl_schedule_node_copy(node_sc);
      return;
    }

    if (band_num > 1) {
      assert (node_ca->n_subgraph == 1);
      
      struct ca_node *next = node_ca->subgraph[0].source->out_edges[path_list->v].dst;
      set_ca_leaf_aux(node_sc, next, path_list->next, idx + 1, band_num - 1);
    }
    else {
      
      for (i = 0; i < node_ca->n_subgraph; ++i) {
	node_sc = isl_schedule_node_child(node_sc, i);

	int n_bands = (isl_schedule_node_get_type(node_sc) == isl_schedule_node_band ?
		       isl_schedule_node_band_n_member(node_sc) : 0);

	set_ca_leaf_aux(node_sc, node_ca->subgraph[i].source, path_list, idx + 1, n_bands);
 
	node_sc = isl_schedule_node_parent(node_sc);
      }
    } // end else
    return;
  }

  assert(path_list != NULL);
  struct ca_node *next = node_ca->out_edges[path_list->v].dst;
  set_ca_leaf_aux(node_sc, next, path_list->next, idx + 1, band_num);
}

void push_kernel_best() {
  struct kern_time_tuple *trav = kernel_best_time_table;
  while(trav != NULL) {
    if (!trav->push) {
      trav->ant = copy_ant(*(trav->ant));
      trav->push = 1;
    }
    trav = trav->next;
  }
}

void post_ca_apply(struct ca_ant *ant, struct coding_ants *ca) {
  printf("post_ca_apply\n");
  //push_kernel_best();
  assert(ca->config.tex_array);
  int i, j;

  if (ant->kernel_metrics) {
    for (i = 0; i < ant->nkernel; ++i) {
      free(ant->kernel_metrics[i]);
    }
    free(ant->kernel_metrics);
  }
  ant->kernel_metrics = NULL;
  ant->nkernel = 0;
  
  /*
  char sched_fname[50];
  const char *pname = get_program_name(ca);
  snprintf(sched_fname, 50, "%s_ref.sched", pname);
  printf("filename = %s\n", sched_fname);
  isl_schedule *sched;
  isl_schedule_node *node;
  
  printf("d1\n");
  sched = load_schedule(ca->ctx, sched_fname);
  node = isl_schedule_get_root(sched);
  printf("d2\n");
  node = isl_schedule_node_child(node, 0);
  printf("d3\n");
  // how many loops does this node represent
  int n_bands = isl_schedule_node_band_n_member(node);
  printf("d4\n");

  //isl_schedule_node_dump(node);
  //printf("n_child %d\n", isl_schedule_node_n_children(node));
  if (n_bands <= 1) {
    printf("d5\n");

    if (isl_schedule_node_get_type(node) == isl_schedule_node_set ||
	isl_schedule_node_get_type(node) == isl_schedule_node_sequence) {
      //node = isl_schedule_node_parent(node);
      printf("n_part = %d\n", ca->n_part);
      printf("children = %d\n", isl_schedule_node_n_children(node));
      //exit(0);
    }
    printf("top children = %d\n", isl_schedule_node_n_children(node));
    for (i = 0; i < ca->n_part; ++i) {
      printf("children = %d\n", isl_schedule_node_n_children(node));
      assert (i < isl_schedule_node_n_children(node));
      node = isl_schedule_node_child(node, i);
      printf("d6\n");
      //isl_schedule_node_dump(node);
      //printf("n_child %d\n", isl_schedule_node_n_children(node));
      //exit(0);
      set_ca_leaf_aux(isl_schedule_node_copy(node), ca->sched_graph[i].source, ant->sched_path[i], 0, 0);
      printf("d7\n");
      node = isl_schedule_node_parent(node);
      printf("d8\n");
    }
  }
  else {
    printf("d9\n");
    assert (n_bands == 2);
    printf("d10\n");
    set_ca_leaf_aux(node, ca->sched_graph[i].source, ant->sched_path[i], 0, n_bands - 1);
    printf("d11\n");
  }
*/
  // clean up
  for (i = 0; i < ant->ntexs; ++i) {
    free(ant->texs[i]);
  }
  free(ant->texs);

  for (i = 0; i < ca->config.n_tex_array; ++i) {
    free(ca->config.tex_array[i]);
  }
  free(ca->config.tex_array);
  ca->config.tex_array = NULL;
  
  //free(ant->sccs);
  ant->sccs = NULL;

  for (i = 0; i < ca->edges->max_dim; ++i) {
    free(ca->config.sccs[i]);
  }
  
  free(ca->config.sccs);
  ca->config.sccs = NULL;

  if (ant->sizes != NULL)
    free(ant->sizes);
  ant->sizes = NULL;

  if (ant->unroll_factors != NULL)
    free(ant->unroll_factors);
  ant->unroll_factors = NULL;

}

void ca_apply_optimizations(struct coding_ants *ca, struct ca_ant *ant, 
			    struct ppcg_options *options) {
  if (1)
    return;

  printf("ca_apply_optimization\n");
  assert(ant->compiler_opts[0].type == COMPILER_OPTION);

  int i, j;
  if (*((int *)ant->compiler_opts[0].val)) { 
    // use shared memory
    options->use_shared_memory = 1;
  }
  else {
    // do not use shared memory
    options->use_shared_memory = 0;
  }

  //for (i = 0; i < ca->edges->max_dim; ++i) {
    //ca->config.sccs[i] = int_copy(ant->sccs[i], ca->n_stmts);
  //}

  // apply memory optimizations   
  int tex_ptr = 0;
  for (i = 0; i < ca->mem_opt_graph.depth; ++i) {
    if(*((int *)ant->mem_opts[i].val)) {
      ca->config.tex_array[tex_ptr++] = strdup(ant->mem_opts[i].name);
    }
  }

  /*char kernel_size[500];
  strcpy(kernel_size, "{");
  for (i = 0; i < ca->n_kernels; ++i) {
    for (j = 0; j < ca->graph[i].depth; ++j) {

      if (ant->opts[i][j].type == CUDA_CONFIG) {
	strcat(kernel_size, ((char *)ant->opts[i][j].val));
	strcat(kernel_size, ";");
      }
    }
  }
  
  strcat(kernel_size, "}");
  */
  #ifdef DEBUG
  printf("sizes = %s\n", kernel_size);
  #endif

  //memset(options->sizes, 0, 1000 * sizeof(char));
  //strcpy(options->sizes, kernel_size);
}

double random_dbl() {
  return (double)rand() / (double)RAND_MAX;
}

float *edge_metrics_mean(struct ca_edge *edge) {
  int i, j;
  float *mets = (float *)calloc(6, sizeof(float));
  
  for (j = 0; j < 6; ++j) {
    int relevant_mets = 0;
    for (i = 0; i < edge->metric_ptr; ++i) {
      if (edge->metrics[i][j] > 0.f)
	++relevant_mets;
    }

    for (i = 0; i < edge->metric_ptr; ++i) {
      if (relevant_mets > 0)
	mets[j] += (edge->metrics[i][j] / relevant_mets);
    }
  }

  return mets;
}

float *edge_metrics_stddev(struct ca_edge *edge) {
  int i, j;
  float *stddev = (float *)calloc(6, sizeof(float));
  float *mean = edge_metrics_mean(edge);
  
  for (j = 0; j < 6; ++j) {
    float sum = 0.0f;
    for (i = 0; i < edge->metric_ptr; ++i) {
      sum += ((edge->metrics[i][j] - mean[j]) * (edge->metrics[i][j] - mean[j]));
    }
    
    stddev[j] = (edge->metric_ptr > 0 ? sqrtf(sum / edge->metric_ptr) : 0.0f);
  }

  free(mean);

  return stddev;
}

/* Compute the eta value for 'edge' based on ant's estimated metric values */
float edge_eta(struct ca_ant *ant, struct ca_edge *edge) {
  int i;
  if (edge->metric_ptr < 3) {
    // this edge has not been traversed
    // return fixed value
    return 1;//0;//0.3;
  }

  float *est = (edge->best_metrics_id < 0 ? ant->est_metrics :
		edge->metrics[edge->best_metrics_id]);//ant->est_metrics;
  
  printf("edge eta estimation = %s, id = %d\n", float_to_string(est, 6), edge->best_metrics_id);
  
  float *mets_mean = edge_metrics_mean(edge);
  float *mets_stddev = edge_metrics_stddev(edge);
    
  float scores[6];
  float score = 0.0f;
  float max_score = -100.0f;
  float min_score = 100.0f;

  for (i = 0; i < 4; ++i) {
    scores[i] = ((mets_mean[i] - est[i]) * (1.0 / (mets_stddev[i] + 1)));
    if (max_score < abs(scores[i]))
      max_score = abs(scores[i]);
  }

  scores[4] = ((mets_mean[4] - est[4]) * (1.0 / (mets_stddev[4] + 1)));
  scores[4] += ((mets_mean[5] - est[5]) * (1.0 / (mets_stddev[5] + 1)));
  scores[4] /= 2;

  if (abs(scores[4]) > max_score)
    max_score = scores[4];

  if (max_score == 0.0f)
    return 1.0;//0.0f;

  for (i = 0; i < 5; ++i) {
    score += (scores[i] / max_score);
  }

  score /= 5;

  /*  if (1) { // multiplicative
    for (i = 0; i < 6; ++i) {
      scores[i] = ((mets_mean[i] - est[i]) * (1.0 / (mets_stddev[i] + 1)));
      if (max_score < (scores[i]))
	max_score = (scores[i]);

      if (min_score > (scores[i]))
	min_score = (scores[i]);
    }
  
    if (max_score - min_score == 0.0)
      return 0.03f;

    for (i = 0; i < 6; ++i) {
      score += ((scores[i] - min_score) / (max_score - min_score));
    }

    score /= 6;
    }*/
  assert (score >= -1 && score <= 1);
  return score + 1.0;
}

/* Update ant's est_metrics with the selected edge's metrics */
void update_ant_metrics(struct ca_ant *ant, struct ca_edge *edge) {
  int i;
  if (1)
    return;

  float *mets_mean = edge_metrics_mean(edge);
  for (i = 0; i < 6; ++i) {
    // compute running average
    float sum = ant->est_metrics[i] * ant->n_est;
    sum += mets_mean[i];
    ant->est_metrics[i] = sum / (ant->n_est + 1);
    //printf("%0.4e\n", ant->est_metrics[i]);
    assert(ant->est_metrics[i] <= 1);
  }

  ++ant->n_est;
}

int select_edge_dfs(struct ca_ant *ant, struct ca_node *node) {
  int i;
  int edge = -1;
  printf("dfs full ex = %d, dfs depth = %d\n", dfs_full_explored, dfs_full_explore_depth);
  if (dfs_full_explored && current_depth == dfs_full_explore_depth - 1) {
    while(dfs_stack_ptr != dfs_full_explore_depth) {
      printf("%d\n", pop());
    }
    //exit(0);
  }

  if (dfs_stack_ptr == current_depth) {
    push(0);
    //dfs_full_explored = (peek() == node->n_edges - 1);
    edge = peek();
  }
  else if(current_depth == dfs_stack_ptr - 1) {
    int e = pop();
    push(e + 1);
    //dfs_full_explored = (peek() == node->n_edges - 1);
    edge = peek();
  }
  else if(current_depth < dfs_stack_ptr) {
    edge = dfs_stack[current_depth];   
  }

  if (edge == node->n_edges -1) {
    if (!dfs_full_explored_n) {
      dfs_full_explored_n = 1;
      dfs_full_explore_depth_n = current_depth;
    }
  }
  else {
    dfs_full_explored_n = 0;
  }

  ++current_depth;
  assert (edge >= 0);
  return edge;
}

int select_edge_random(struct ca_ant *ant, struct ca_node *node) {
  int i;
  double r = random_dbl();
  double total_weight = 0;
  double ew = 1.0 / ((double)node->n_edges);

  // if an edge hasn't been taken before, then take it
  for (i = 0; i < node->n_edges; ++i) {
    if (!node->out_edges[i].has_been_traversed) {
      is_new_sample = 1;
      node->out_edges[i].has_been_traversed = 1;
      push(i);
      return i;
    }
  }

  for (i = 0; i < node->n_edges; ++i) {
    total_weight += ew;
    if (r <= total_weight) {
      node->out_edges[i].has_been_traversed = 1;
      push(i);
      return i;
    }
  }
  printf("BAD\n");
  exit(0);
  return node->n_edges - 1;
}

/* Select the edge to follow probablisitically based on 
 * the pheremone values on each edge. 
 * Return the index of the selected edge
 */
int select_edge(struct ca_ant *ant, struct ca_node *node, int *blacklist) {
  int i;
  
  if (depth_first_search) {
    int e =  select_edge_dfs(ant, node);
    printf("e = %d\n", e);
    printf("dfs = %d\n", dfs_stack_ptr);
    assert (e < node->n_edges);
    return e;
  }

  if (random_search) {
    return select_edge_random(ant, node);
  }

  if (choose_best) {
    int e = -1;
    double best_score;
    for (i = 0; i < node->n_edges; ++i) {
      struct ca_node *next = node->out_edges[i].dst;

      if (next) {
	double score;
	if (next->in_kernel) {
	  score = (double) next->kernel_score;	  
	}
	else {
	  score = next->best_score;
	}

	if (e < 0 || score < best_score) {
	  e = i;
	  best_score = score;
	}
      }      
    } // end for i

    //assert (e < node->n_edges);
    //if (e > 0)
    printf("e = %d, nedges = %d, node = %s\n", e, node->n_edges, node->name);
    assert (e >= 0);
    return e;
  }

  if (best_first_search) {
    ++current_depth;
    printf("HERE\n");
    printf("%d, %d\n", current_depth, beam_depth);
    
    if (node == beam_node) {
      printf("NODE == BEAM, beam edge = %d\n", beam_edge);
      assert (current_depth == beam_depth);

      if (blacklist) {
	n_possible_edges = 0;
	for (i = 0; i < node->n_edges; ++i) {
	  if (!blacklist[i]) {
	    ++n_possible_edges;
	  }
	}
      }

      if (blacklist) {
	for (i = beam_edge; i < node->n_edges; ++i) {
	  if (blacklist[i] == 0) {
	    beam_edge = i;
	    printf("edged = %d\n", beam_edge);
	    return beam_edge;
	  }
	}
      }

      printf("edge = %d\n", beam_edge);
      return beam_edge;
    }
        
    if (current_depth < beam_depth) {
      int e = current_path->v;
      printf("edge = %d\n", e);
      assert(node->out_edges[e].dst);
      current_path = current_path->next;

      if (blacklist) {
	printf("%s\n", int_to_string(blacklist, node->n_edges));
	assert (blacklist[e] == 0);
      }
      return e;
    }
    
    if (blacklist) {
      for (i = 0; i < node->n_edges; ++i){
	if (blacklist[i] == 0){
	  if (current_depth == beam_depth+1) {
	    current_bfs_next_edge = i;
	  }
	  return i;
	}	  
      }
    }
    
    if (current_depth == beam_depth+1) {
      current_bfs_next_edge = 0;
    }    
    return 0;
  } // end best_first_search

  //if (1)
  //return node->n_edges - 1;
  //if (1) 
  //return 0;
  // get a random number
  double r = random_dbl();

  double sum_weight = 0.0;

  int not_all_blocked = 0; 
  int n_edges = node->n_edges;
  if (blacklist) {
    printf("lb = %s\n", int_to_string(ant->filter.attr_lb, num_regression_vec));
    printf("ub = %s\n", int_to_string(ant->filter.attr_ub, num_regression_vec));
   
    // at least one edge should be available
    for (i = 0; i < n_edges; ++i) {
      if (!blacklist[i]) {
	not_all_blocked = 1;
	break;
      }
    }
    if (!not_all_blocked) {
      printf("NOT ALL BLOCKED\n");
      // undo the blacklist
      //for (i = 0; i < n_edges; ++i)
      //blacklist[i] = 0;
    }
    //assert (not_all_blocked);
  }

  double total_eta = 0.0f;
  // normalize by the sum of weights to one
  double total_weight = 0;
  for (i = 0; i < n_edges; ++i) {
    if (blacklist && blacklist[i] && not_all_blocked) {
      continue;
    }

    float eta = edge_eta(ant, &node->out_edges[i]);
    double tau = node->out_edges[i].tau;    
    //total_weight += max((tau * ALPHA) + (eta * BETA), 0.01);
    total_weight += pow(tau, ALPHA) * pow(eta, BETA);
    //total_weight += tau;
    printf("edge %d: tau = %0.4e, eta = %0.4e, traversed = %d\n", i, 
	   (pow(tau, ALPHA)), (pow(eta, BETA)), node->out_edges[i].has_been_traversed);


    if (1) {
      assert (tau == tau); // nan check
      assert (tau >= 0);
    }

  }

  printf("%f\n", total_weight);
  assert (total_weight > 0);
  for (i = 0; i < n_edges; ++i) {
    if (blacklist && blacklist[i] && not_all_blocked)
      continue;

    float eta = edge_eta(ant, &node->out_edges[i]);
    double tau = node->out_edges[i].tau;
    
    //sum_weight += (node->out_edges[i].tau / total_weight);
    //sum_weight += (max((tau * ALPHA) + (eta * BETA), 0.01) / total_weight);
    sum_weight += ((pow(tau, ALPHA) * pow(eta, BETA)) / total_weight);
    if (sum_weight >= r) {
      printf("selected edge %d r = %0.4e\n\n", i, r);
      node->out_edges[i].has_been_traversed = 1;
      //decay_edge(&node->out_edges[i]);      
      update_ant_metrics(ant, &node->out_edges[i]);
      return i;
    }
  }

  printf("selected edge %d r = %0.4e\n\n", (n_edges - 1), r);
  node->out_edges[n_edges - 1].has_been_traversed = 1;

  update_ant_metrics(ant, &node->out_edges[n_edges - 1]);
  //decay_edge(&node->out_edges[n_edges - 1]);  
  return n_edges - 1;
}

void update_scc_edges(struct simple_edges *graph, const int scc) {
  struct scc_edge *trav = graph->node[scc].edges;

  while(trav->dst != NULL) {
    if (trav->dst->n_unsat > 0)
      --trav->dst->n_unsat;
    //assert(trav->dst->n_unsat >= 0);
    trav = trav->next;
  }

  // mark this node as already being placed
  graph->node[scc].n_unsat = -1;
}

void ca_traverse_partition_size(struct ca_ant *ant, struct coding_ants *ca, int rem_part, 
				int num_sccs, 
				struct ca_node *node, struct path *path_list, 
				int *idx, int *part_sizes, int *part_id, int **scc_part,
				const int lev) {
 
  if (rem_part == 0) {
    // base case
    return;
  }

  // select a path
  const int e_idx = select_edge(ant, node, NULL);
  path_list->v = e_idx;
  path_list->next = (struct path *)calloc(1, sizeof(struct path)); 
  path_list->next->next = NULL;
  (*idx)++;

  //path[(*idx)++] = e_idx;
  
  const int part_size = (rem_part > 1 ? e_idx + 1 : num_sccs);
  part_sizes[(*part_id)++] = part_size;

  struct ca_node *next = NULL;

  // e_idx + 1 is the partition size
  if(node->out_edges[e_idx].dst == NULL) {    
    char buf[50];
    snprintf(buf, 50, "part_size:%d", part_size);

    int *v = (int *)calloc(1, sizeof(int));
    *v = part_size;
    
    next = create_node(PART_SIZE, buf, v);
    int n_edges = (num_sccs - part_size) - (rem_part - 1) + 1;
    
    node->out_edges[e_idx].dst = next;    
    
    if (rem_part > 1) {
      ca_allocate_out_edges(next, (rem_part - 1 > 1 ? n_edges : 1));
    }
  }
  else {
    next = node->out_edges[e_idx].dst;
    if (rem_part == 1 && lev + 1 < ca->edges->max_dim) {
      assert (next->aux != NULL);
      printf("at seen leaf\n");
      // we've seen this leaf node 
      isl_schedule_node *sch = (isl_schedule_node *)next->aux;
      
      const int n_part = isl_schedule_node_n_children(sch);
      
      reset_graph(ca->edges);
      // update scc graph to only include sccs in the partition
      int n_scc = 0;
      int *node_sccs = get_sccs_from_stmts(ca->edges, ca->n_stmts, sch, &n_scc);
      
      isl_schedule_node_dump(sch);

      printf("f1\n");
      int i;
      for (i = 0; i < ca->edges->n_scc; ++i) {
	if (!int_exists(ca->edges->node[i].scc, node_sccs, n_scc)) {
	  // the scc is in some other partition
	  // update the scc graph
	  update_scc_edges(ca->edges, ca->edges->node[i].scc);
	  printf("doesnt contain scc %d\n", ca->edges->node[i].scc);
	}	
      }

      for (i = 0; i < ca->edges->n_scc; ++i) {
	printf("scc %d, n_unsat = %d, n_in = %d\n", ca->edges->node[i].scc, 
	       ca->edges->node[i].n_unsat, ca->edges->node[i].n_in);
	
      }

      printf("f2\n");

      if (next->n_subgraph == 0) {
	printf("subgraph is zero\n");


	if (n_part > ca->edges->n_scc) {
	  // stick with pluto for now
	  return;
	}
	next->n_subgraph = n_part;
	next->subgraph = (struct ca_graph *)calloc(n_part, sizeof(struct ca_graph));     
	
	build_root(sch, next->subgraph, ca->edges, n_part, ca->n_stmts);
	reset_graph(ca->edges);
      }
      
      int start_part = 0;

      // for now
      //assert (n_part == 1);

      for (i = 0; i < n_part; ++i) {
	int sched_idx = 0;
	scc_part = ca_traverse_sched_graph(ant, ca, next->subgraph[i].source, path_list->next, 
					   &sched_idx, scc_part, &start_part, lev + 1);
	
	printf("DEBUG----------------\n");
	struct ca_node *tmp = next->subgraph[i].source;
	struct path *tmp_path = path_list->next;
	
	while(tmp->out_edges != NULL) {
	  printf("%s\n", tmp->name);
	  tmp = tmp->out_edges[tmp_path->v].dst;
	  tmp_path = tmp_path->next;
	}
	printf("%s\n", tmp->name);
	printf("END DEBUG----------------\n");
	// update the path ptr
	path_list = get_tail(path_list);
      }
      
      
      for (i = 0; i <= lev + 1; ++i) {
	printf("sccs at %d = %s\n", i, int_to_string(scc_part[i], ca->n_stmts));
      }

      print_graph_to_file("tst.graph", next->subgraph[0].source);

      isl_schedule_node_dump(sch);
      printf("n_child %d\n", isl_schedule_node_n_children(sch));
      printf("seen leaf\n");

      //if (lev > 0)
      //	exit(0);   
    }
  }

  printf("%s --> %s\n", node->name, next->name);
  ca_traverse_partition_size(ant, ca, rem_part - 1, num_sccs - part_size, next, path_list->next, 
			     idx, 
			     part_sizes, part_id, scc_part, lev);
}

// nodes: an array of scc_nodes
// scc_order: an array of scc ids in the selected order
// part_sizes: array of partition sizes
// n_part: number of partitions (i.e. length of the part_sizes array)
// out: array of integers to be populated with partition ids. The positions
// correspond to stmts.
void get_stmt_order(struct scc_node *nodes, int *scc_order, int *part_sizes, 
		    const int n_part, int *out, int *start_part) {
  int i, j, k;
  
  int out_idx = 0;

  // for each partition
  for (i = 0; i < n_part; ++i) {
    for (j = 0; j < part_sizes[i]; ++j) {
      struct scc_node *scc = &nodes[scc_order[out_idx++]];
      // for each stmt in the scc
      for (k = 0; k < scc->n; ++k) {
	out[scc->stmts[k]] = *start_part;
      } // end for k
    } // end for j
    ++(*start_part);
  } // end for i
}

int **ca_traverse_partitioning(struct ca_ant *ant, struct coding_ants *ca, struct ca_node *node, 
			      struct scc_vec *vec, int *scc_order, struct path *path_list, 
			       int *idx, int **scc_part, int *start_part, const int lev) {
  
  assert (node->type == SCC);
  
  struct simple_edges *edges = ca->edges;

  // select a path
  const int e_idx = select_edge(ant, node, NULL);
  path_list->v = e_idx;
  path_list->next = (struct path *)calloc(1, sizeof(struct path));
  path_list->next->next = NULL;
  (*idx)++;

  //path[(*idx)++] = e_idx;

  // there can be between 1 and num_sccs partitons
  // e_idx + 1 is the number of partitions
  int n_part = e_idx + 1;

  struct ca_node *next = NULL;

  if(node->out_edges[e_idx].dst == NULL) {

    char buf[50];
    snprintf(buf, 50, "n_part:%d", n_part);

    int *v = (int *)calloc(1, sizeof(int));
    *v = n_part;

    next = create_node(NUM_PARTS, buf, v);  
    const int max_size = vec->len - n_part + 1;
      

    ca_allocate_out_edges(next, (n_part > 1 ? max_size : 1));
      
    node->out_edges[e_idx].dst = next;  
  }
  else {
    next = node->out_edges[e_idx].dst;   
  }

  printf("%s --> %s\n", node->name, next->name);  
  int *part_sizes = (int *)calloc(n_part, sizeof(int));
  if (scc_part[lev] == NULL) {
    scc_part[lev] = (int *)calloc(ca->n_stmts, sizeof(int));
  }
  printf("d1 lev = %d\n", lev);
  int part_id = 0;
  ca_traverse_partition_size(ant, ca, n_part, vec->len, next, path_list->next, idx, part_sizes, 
			     &part_id, scc_part, lev);
  printf("d2 lev = %d\n", lev);
  //int *sccs = (int *)calloc(vec->len, sizeof(int));  
  get_stmt_order(edges->node, scc_order, part_sizes, n_part, scc_part[lev], start_part);
  printf("d3 lev = %d\n", lev);
  return scc_part;  
}

int **ca_traverse_sched_order(struct ca_ant *ant, struct coding_ants *ca, struct ca_node *node, 
			     struct scc_vec *vec, struct path *path_list, int *idx, 
			     int *scc_order, int **scc_part, int *start_part, const int lev) {
  

  struct simple_edges *edges = ca->edges;
  if ((*idx) == vec->len) {
    if (node->out_edges == NULL) {
      // prepare for the number of partitions
      ca_allocate_out_edges(node, vec->len);
    }

    return ca_traverse_partitioning(ant, ca, node, vec, scc_order, path_list, idx, 
				    scc_part, start_part, lev);
  }
  
  const int e_idx = select_edge(ant, node, NULL);
  path_list->v = e_idx;
  path_list->next = (struct path *)calloc(1, sizeof(struct path)); 
  path_list->next->next = NULL;
  (*idx)++;

  //path[(*idx)++] = e_idx;

  int n_valid = 0;
  // number of sccs with 0 unsatisfied dependences
  int *valid = get_valid_sccs(edges, &n_valid, vec->sccs, vec->len);

  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char buf[50];
    snprintf(buf, 50, "scc:%d", valid[e_idx]);
    
    int *v = (int *)calloc(1, sizeof(int));
    *v = valid[e_idx];

    scc_order[(*idx) - 1] = *v;
    
    next = create_node(SCC, buf, v);  

    // update the predecessor edges
    update_scc_edges(edges, valid[e_idx]);
    n_valid = get_num_valid_sccs(edges, vec->sccs, vec->len);

    ca_allocate_out_edges(next, n_valid);
      
    node->out_edges[e_idx].dst = next;
  } 
  else {    
    next = node->out_edges[e_idx].dst;
    scc_order[(*idx) - 1] = (*((int *)next->val));
    update_scc_edges(edges, scc_order[(*idx) - 1]);    
  }
  
  printf("%s ---> %s\n", node->name, next->name);
  ca_traverse_sched_order(ant, ca, next, vec, path_list->next, idx, scc_order, scc_part, 
			  start_part, lev);
}
// return a copy vector of scc partition 
int **ca_traverse_sched_graph(struct ca_ant *ant, struct coding_ants *ca, struct ca_node *node, 
			     struct path *path_list, int *idx, 
			     int **scc_part, int *start_part, const int lev) {

  assert(node->type == SOURCE);
  struct scc_vec *vec = (struct scc_vec *)node->val;
  
  // a vector of scc ids to be populated
  int *scc_order = (int *)calloc(vec->len, sizeof(int));
  return ca_traverse_sched_order(ant, ca, node, vec, path_list, idx, scc_order, 
				 scc_part, start_part, lev);  
}

void ca_traverse_graph(struct ca_ant *ant, struct ca_node *opts, 
		       int *path, struct ca_graph graph) {
  int i;

  struct ca_node *node = graph.source;

  int depth = graph.depth;
  for (i = 0; i < depth; ++i) {
    const int e_idx = select_edge(ant, node, NULL);

    // decay pheremone value on this edge to 
    // encourage more random search by other ants
    decay_edge(&node->out_edges[e_idx]);

    node = node->out_edges[e_idx].dst;

    opts[i] = *node;          
    path[i] = e_idx;
  }

  // reporting
  for (i = 0; i < graph.depth; ++i) {
    print_val(opts[i]);
  }
  printf("\n");    
}

struct ca_ant *init_ants(struct coding_ants *ca, int nread_only) {
  int i, j;
  assert(ca->n_stmts > 0);
  int optimization = ca->wrap->optimization;
  num_regression_vec = 
    (optimization > 0 ? ca->n_stmts : 0) // partition
    + (optimization == 2 ? ca->n_stmts : 0) // outer skewing
    + 1 // reduction 
    + nread_only // texture arrays
    + ca->n_stmts // tile coarseness
    + ca->n_stmts // inner tile
    + ca->n_stmts; // unrolling

  unroll_start = num_regression_vec - ca->n_stmts;
  inner_tile_start = unroll_start - ca->n_stmts;  
  tile_start = inner_tile_start - ca->n_stmts;
  texture_start = tile_start - nread_only;
  reduction_start = texture_start - 1;
  if (optimization == 0) {
    skewing_start = -1;
    partition_start = -1;
  }
  else {
    skewing_start = reduction_start - ca->n_stmts;
    partition_start = (optimization == 2 ? 
		       skewing_start - ca->n_stmts :
		       reduction_start - ca->n_stmts);
    assert (partition_start == 0);
  }

  struct ca_ant *ants = (struct ca_ant*)calloc(NUM_ANTS, sizeof(struct ca_ant));
  for (i = 0; i < NUM_ANTS; ++i) {
    // allocate memory opt path
    ants[i].mem_opts = (struct ca_node*)calloc(ca->mem_opt_graph.depth, sizeof(struct ca_node));
    ants[i].mem_opt_path = (int *)calloc(ca->mem_opt_graph.depth, sizeof(int));
    
    ants[i].s_path = (struct path *)calloc(1, sizeof(struct path));
    ants[i].s_path->next = NULL;

    ants[i].sizes = NULL;
    ants[i].unroll_factors = NULL;
    
    ants[i].kernel_stmts = NULL;
    ants[i].kernel_stmt_id = 0;

    if (optimization == 0) {
      ants[i].compiler_options = (int *)calloc(4, sizeof(int));
    }

    ants[i].est_metrics = (float *)calloc(6, sizeof(float));

    ants[i].opt_vec = (int *)calloc(num_regression_vec, sizeof(int));
    //ants[i].filter.attr_vec = (int *)calloc(num_regression_vec, sizeof(int));
    ants[i].filter.attr_lb = (int *)calloc(num_regression_vec, sizeof(int));
    ants[i].filter.attr_ub = (int *)calloc(num_regression_vec, sizeof(int));
    //ants[i].filter.lte_mask = (int *)calloc(num_regression_vec, sizeof(int));
    
    /*    assert (ca->n_part > 0);
    ants[i].sched_path = (struct path **)calloc(ca->n_part, sizeof(struct path *));
    for (j = 0; j < ca->n_part; ++j) {
      assert(ca->sched_graph[j].depth > 0);
      ants[i].sched_path[j] = NULL;
      ants[i].sched_path[j] = (struct path *)calloc(1, sizeof(struct path));
      ants[i].sched_path[j]->next = NULL;
      //ants[i].sched_path[j] = (int *)calloc(ca->sched_graph[j].depth, sizeof(int));
      assert(ants[i].sched_path[j] != NULL);
    }
    */
    //ants[i].opts = (struct ca_node**)calloc(ca->n_kernels, sizeof(struct ca_node *));
    //ants[i].opt_path = (int **)calloc(ca->n_kernels, sizeof(int *));
    //for (j = 0; j < ca->n_kernels; ++j) {
    //  ants[i].opts[j] = (struct ca_node*)calloc(ca->graph[j].depth, sizeof(struct ca_node));
    //  ants[i].opt_path[j] = (int *)calloc(ca->graph[j].depth, sizeof(int));
    //}
  }

  return ants;
}

int generate_code(isl_ctx *ctx,  struct ppcg_options *options,
	       const char *input, const char *output, 
	       __isl_give isl_printer *(*print_cuda)(__isl_take isl_printer *p,
		struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
		struct gpu_types *types, void *user),struct coding_ants *ca) {
  
  printf("generate_code\n");
  struct cuda_info cuda;
  cuda_open_files(&cuda, strdup(input), strdup(output));
  cuda.wrap = ca->wrap;
  int r = generate_gpu(ctx, strdup(input), cuda.host_c, options, print_cuda, &cuda, ca);  
  cuda_close_files(&cuda);
  printf("end generate_code\n");
  return r;

}

int get_scc_from_stmt(const int stmt, struct simple_edges *graph) {
  int i, j;
  for (i = 0; i < graph->n_scc; ++i) {
    for (j = 0; j < graph->node[i].n; ++j) {
      if (stmt == graph->node[i].stmts[j]) {
	return graph->node[i].scc;
      }
    }
  }

  return -1;
}

int get_scc_pos(int scc, int *scc_order, struct simple_edges *graph) {
  int i;
  for (i = 0; i < graph->n_scc; ++i) {
    if (scc_order[i] == scc) {
      return i;
    }
  }

  return -1;
}

int *get_partition_size_filter(struct ca_ant *ant, struct simple_edges *graph, 
			       int *scc_order, const int n_edges, 
			       const int current_partition, const int sum_partitions) {
  int i, j;
  const int n_stmts = graph->n_stmts;
  if (!ant->first_partition || !ant->filter_set)
    return NULL;
    
  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  for (i = 0; i < n_stmts; ++i) {
    //int split = ant->filter.attr_vec[i];
    int idx = graph->stmt_ids[i];
    int lb = ant->filter.attr_lb[idx];
    int ub = ant->filter.attr_ub[idx];

    int scc = get_scc_from_stmt(i, graph);
    int scc_pos = get_scc_pos(scc, scc_order, graph);
    assert (scc_pos >= 0);

    if (lb >= 0) {
      lb -= ant->opt_vec[idx];
      // must be >= lb
      if (current_partition < lb) {	  
	int max_size = scc_pos - lb + 1 - sum_partitions + current_partition;
	for (j = max_size; j < n_edges; ++j) {
	  blacklist[j] = 1;
	}
      }	
    }

    if (ub >= 0) {
      ub -= ant->opt_vec[idx];
      if (sum_partitions < scc_pos && current_partition == ub) {
	// last chance to set scc to partition 'split'
	int min_size = scc_pos - sum_partitions;
	printf("min size = %d, n_edges = %d, scc_pos = %d\n", min_size, n_edges, scc_pos);
	//assert (min_size < n_edges);
	if (min_size < n_edges) {
	  for (j = 0; j < min_size; ++j)
	    blacklist[j] = 1;
	}

      }
    }
  }

  return blacklist;
}

int *add_part_size_depth_cstr(int *blacklist, const int n_edges, int *scc_order, 
			      struct simple_edges *graph, const int sum_part,
			      const int vsrc, const int vdst) {
  int i;
  if (vsrc < 0) {
    return blacklist;
  }
  
  assert (vdst >= 0);
  if (!blacklist) {
    blacklist = (int *)calloc(n_edges, sizeof(int));
  }

  int src_pos = get_scc_pos(vsrc, scc_order, graph);
  int dst_pos = get_scc_pos(vdst, scc_order, graph);
  printf("%s\n", int_to_string(scc_order, graph->n_scc));
  printf("src = %d, dst = %d\n", vsrc, vdst);
  assert(src_pos >= 0);
  assert(dst_pos >= 0);
  
  assert (src_pos < dst_pos);
  
  if (sum_part <= src_pos) {
    for (i = 0; i < n_edges; ++i) {
      int size = i + 1;
      if (sum_part + size > dst_pos) {
	blacklist[i] = 1;
      }
    } // end for i
  }
  
  return blacklist;
}

/* Select the size of each partition.
 * ant: the traversing ant
 * node: the current node the ant is at
 * n_scc: the number of sccs that have not been 
 * assigned a partition
 * rem_part: the number of remaining partitions
 * p_sizes: a vector of partition sizes
 * pid: idx into the p_sizes array
 * return: a vector of partition sizes
 */
int *ca_traverse_partition_sizes(struct ca_ant *ant, struct path *s_path, struct ca_node *node,
				 const int n_scc, int rem_part, int *p_sizes, int *pid, 
				 struct simple_edges *graph, int *scc_order, 
				 const int vsrc, const int vdst) {
  int i;
  if (rem_part == 0) { // base case
    // no more remaining partitions
    return p_sizes;
  }
  
  int sum_part = 0;
  for (i = 0; i < *pid; ++i)
    sum_part += p_sizes[i];

  int *filter = (rem_part > 1 && node->n_edges > 1 ? 
		 get_partition_size_filter(ant, graph, scc_order, node->n_edges, *pid, sum_part) :
		 NULL);

  filter = (rem_part > 1 && node->n_edges > 1 ? 
	    add_part_size_depth_cstr(filter, node->n_edges, scc_order, graph,
				     sum_part, vsrc, vdst) :
	    filter);
  
  // select a path
  const int e_idx = select_edge(ant, node, filter);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;

  if (filter) {
    printf("partisze filt = %s\n", int_to_string(filter, node->n_edges));
    free(filter);
    filter = NULL;
  }

  const int part_size = (rem_part > 1 ? e_idx + 1 : n_scc);

  // add to vector of partition sizes
  p_sizes[(*pid)++] = part_size;

  struct ca_node *next = NULL;
    // e_idx + 1 is the partition size
  if(node->out_edges[e_idx].dst == NULL) {    
    char buf[50];
    snprintf(buf, 50, "part_size:%d", part_size);

    int *v = (int *)calloc(1, sizeof(int));
    *v = part_size;
    
    next = create_node(PART_SIZE, buf, v);
    next->stmt_len = node->stmt_len;
    next->stmt_str = node->stmt_str;

    // we have 'part_size' fewer sccs and 1 fewer partition
    int n_edges = (n_scc - part_size) - (rem_part - 1) + 1;
    
    node->out_edges[e_idx].dst = next;    
    
    if (rem_part > 1) {
      ca_allocate_out_edges(next, (rem_part - 1 > 1 ? n_edges : 1));
    }
  }
  else {
    next = node->out_edges[e_idx].dst;
    printf("next %d, psize %d\n", (*(int *)next->val), part_size);
    assert((*(int *)next->val) == part_size);
  }

  return ca_traverse_partition_sizes(ant, s_path->next, next, n_scc - part_size, 
				     rem_part - 1, p_sizes, pid, graph, scc_order, 
				     vsrc, vdst);
}

int *get_num_partition_filter(struct ca_ant *ant, struct simple_edges *graph, 
			      int *scc_order, const int n_edges) {
  int i, j;
  const int n_stmts = graph->n_stmts;
  if (!ant->first_partition || !ant->filter_set)
    return NULL;
  
  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  for (i = 0; i < n_stmts; ++i) {
    int idx = graph->stmt_ids[i];
    int lb = ant->filter.attr_lb[idx]; 
    int ub = ant->filter.attr_ub[idx];
    
    if (lb >= 0) {
      lb -= ant->opt_vec[idx];
      // must be >= lb
      //assert (lb <= n_edges);
      
      // number of partitions must be
      // at least 'lb'
      for (j = 0; j < lb; ++j) {
	if (j >= n_edges - 1)
	  break;

	blacklist[j] = 1;
      }
    }

    if (ub >= 0) {
      // must be <= ub
      ub -= ant->opt_vec[idx];
      int scc = get_scc_from_stmt(i, graph);
    
      // npart <= nsccs - sccs_pos - 1
      int scc_pos = get_scc_pos(scc, scc_order, graph);
      assert (scc_pos >= 0);

      // maximum number of partitions - 1
      int n_part = graph->n_scc - scc_pos - 1 + ub;	
      printf("npart = %d\n", n_part + 1);
      for (j = n_part + 1; j < n_edges; ++j) {
	blacklist[j] = 1;
      }
    }
  } // end for i
  
  return blacklist;
}

/* Add constraint to prevent vsrc and vdst from being in the same partition */
int *add_num_partition_depth_cstr(int *blacklist, const int n_edges, 
				  const int vsrc, const int vdst) {
  int i;
  if (vsrc == -1)
    return blacklist;

  if (vsrc == -2) {
    // flag for max fusion
    if (!blacklist) {
      blacklist = (int *)calloc(n_edges, sizeof(int));
    }
    for (i = 1; i < n_edges; ++i) {
      blacklist[i] = 1;
    }
    return blacklist;
  }
  
  assert (vdst >= 0);

  if (!blacklist) {
    blacklist = (int *)calloc(n_edges, sizeof(int));
  }

  // avoid the max fuse solution
  blacklist[0] = 1;
  return blacklist;
}

/* Select the number of partitions and call the function 
 * to set the size of each partition.
 * ant: the traversing ant
 * node: the current node the ant is at
 * graph: the ca representation of the scc graph
 * scc_order: array of sccs selected so far
 * return: array indicating a partitioning of sccs
 */
int *ca_traverse_num_partition(struct ca_ant *ant, struct path *s_path, struct ca_node *node,
			       struct simple_edges *graph, int *scc_order, 
			       const int vsrc, const int vdst) {
  int i, j, k; 
  assert(node->type == SCC);
  assert(node->out_edges != NULL);
  assert(node->n_edges > 0);

  int *filter = (node->n_edges > 1 ? 
    get_num_partition_filter(ant, graph, scc_order, node->n_edges) :
    NULL);;

  printf("numpart filt = %s\n", int_to_string(filter, node->n_edges));
  filter = (node->n_edges > 1 ? 
	    add_num_partition_depth_cstr(filter, node->n_edges, vsrc, vdst) :
	    NULL);
  printf("numpart filt2 = %s\n", int_to_string(filter, node->n_edges));
  printf("vsrc = %d, vdst = %d\n", vsrc, vdst);
  // select a path
  const int e_idx = select_edge(ant, node, filter);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  
  if (filter) {
    printf("numpart filt3 = %s\n", int_to_string(filter, node->n_edges));
    //free(filter);
    printf("finish free\n");
    filter = NULL;
    printf("filter = null\n");
    
  }

  // there can be between 1 and num_sccs partitons
  // e_idx + 1 is the number of partitions
  const int n_part = e_idx + 1;

   struct ca_node *next = NULL;
  if(node->out_edges[e_idx].dst == NULL) {
    char buf[50];
    snprintf(buf, 50, "n_part:%d", n_part);

    int *v = (int *)calloc(1, sizeof(int));
    *v = n_part;

    next = create_node(NUM_PARTS, buf, v);  
    next->stmt_len = node->stmt_len;
    next->stmt_str = node->stmt_str;
    // number of possible partition sizes for the 
    // first partition
    const int n_first_part_size = (n_part > 1 ? 
				   (graph->n_scc - n_part + 1) : 1);
        
    ca_allocate_out_edges(next, n_first_part_size);    
    node->out_edges[e_idx].dst = next;  
  }
  else {
    next = node->out_edges[e_idx].dst;   
  }

  // vector of partition sizes
  int *p_sizes = (int *)calloc(n_part, sizeof(int));
  int pid = 0; // idx into p_sizes

  printf("n_part = %d\n", n_part);
  p_sizes = ca_traverse_partition_sizes(ant, s_path->next, next, graph->n_scc, 
					n_part, p_sizes, &pid, graph, scc_order, vsrc, vdst);  
  printf("p_sizes = %s\n", int_to_string(p_sizes, n_part));
  
  // set the partitioning array mapping statements to partition id
  int *sccs = (int *)calloc(graph->n_stmts, sizeof(int));
  int scc_idx = 0;
  for (i = 0; i < n_part; ++i) {
    for (j = 0; j < p_sizes[i]; ++j) {    
      // current scc
      int c_scc = scc_order[scc_idx++];
      // for each stmt in the scc
      for (k = 0; k < graph->node[c_scc].n; ++k){
	sccs[graph->node[c_scc].stmts[k]] = i;
      }
    }
  }
  
  if (ant->first_partition) {
    // get the max scc num
    int max_num = 0;
    for (i = 0; i < graph->n_stmts; ++i) {
      max_num = (max_num < sccs[i] ? sccs[i] : max_num);
    }

    int max_part = 0;
    for (i = 0; i < graph->n_stmts; ++i) {
      max_part = (max_part < ant->opt_vec[graph->stmt_ids[i] + partition_start] ?
		  ant->opt_vec[graph->stmt_ids[i] + partition_start] : max_part);
    }
    

    // shift predecessor partitions by the max num
    int n_total_stmts = reduction_start - skewing_start; //partition_start;
    for (i = 0; i < n_total_stmts; ++i) {
      if (ant->opt_vec[i + partition_start] > max_part) {
	ant->opt_vec[i + partition_start] += max_num;
      }
    }

    for (i = 0; i < graph->n_stmts; ++i) {
      //ant->opt_vec[ant->opt_vec_ptr++] = sccs[i];
      ant->opt_vec_ptr++;
      ant->opt_vec[graph->stmt_ids[i] + partition_start] += sccs[i];// + graph->partition_start;
    }
    
    ant->first_partition = 0;
    printf("%s\n", int_to_string(sccs, graph->n_stmts));
    printf("%s\n", int_to_string(graph->stmt_ids, graph->n_stmts));
    printf("%s\n", int_to_string(ant->opt_vec, graph->n_stmts));
  }

  // cleanup
  free(p_sizes);
  printf("sccs %s\n", int_to_string(sccs, graph->n_scc));
  return sccs;
}

int *get_scc_order_filter(struct ca_ant *ant, struct simple_edges *graph, 
			  int *valid, int len, int order_idx) {
  int i, j;
  const int n_stmts = graph->n_stmts;
  //if (ant->opt_vec_ptr >= n_stmts) {
  //  return NULL;
  //}
  
  if (!ant->first_partition || !ant->filter_set) {
    return NULL;
  }
  
  // mask w/size = number of edges. if the value
  // is 1 then prevent the edge from being selected
  int *blacklist = (int *)calloc(len, sizeof(int));

  for (i = 0; i < n_stmts; ++i) {
    int idx = graph->stmt_ids[i];
    int lb = ant->filter.attr_lb[idx];
    
    // we don't consider the ub here because we 
    // can always perform a max fuse if necessary
    if (lb >= 0) {
      // filter is attr >= lb
      int scc = get_scc_from_stmt(i, graph);
      assert (scc >= 0);
      
      printf("idx = %d, order idx = %d, opt_vec = %d\n", idx, order_idx, ant->opt_vec[idx]);
      // postition of scc must be >= lb
      if (order_idx + ant->opt_vec[idx] < lb) {
	// add idx in valid that corresponds to
	// scc to the blacklist.
	for (j = 0; j < len; ++j) {
	  if (valid[j] == scc) {
	    blacklist[j] = 1;
	  }
	} // end for j
      }

    }
  } // end for i

  return blacklist;
}

/* Recursively select a scc order and call the function to 
 * select the number of partitions.
 * ant: the traversing ant
 * node: the current node the ant is at
 * graph: the ca representation of the scc graph
 * scc_order: array of sccs selected so far
 * order_idx: idx into the scc_order array
 * return: array indicating a partitioning of sccs
 */
int *ca_traverse_statement_order(struct ca_ant *ant, struct path *s_path, struct ca_node *node,
				 struct simple_edges *graph, int *scc_order, int *order_idx,
				 const int vsrc, const int vdst) {

  int n_valid = get_num_valid_sccs2(graph);
  if (n_valid == 0) { // base case
    if (node->out_edges == NULL) {
      // prepare for number of partitions
      ca_allocate_out_edges(node, graph->n_scc);
    }
    
    printf("scc order = %s\n", int_to_string(scc_order, graph->n_scc));
    
    // select number of partitions
    return ca_traverse_num_partition(ant, s_path, node, graph, scc_order, vsrc, vdst);
  }
  assert(node->out_edges != NULL);

  int *valid = get_valid_sccs2(graph);
  
  int *filter = get_scc_order_filter(ant, graph, valid, n_valid, *order_idx);  
  const int e_idx = select_edge(ant, node, filter);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  
  if (!ant->first_partition) {
    // update opt_vec for regression tree
    //ant->opt_vec[ant->opt_vec_ptr++] = valid[e_idx] + 1;
  }

  update_scc_edges(graph, valid[e_idx]);  
  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char buf[50];
    snprintf(buf, 50, "scc:%d", valid[e_idx]);
    
    int *v = (int *)calloc(1, sizeof(int));
    *v = valid[e_idx];
    
    // create the next node
    next = create_node(SCC, buf, v);    
    next->stmt_len = node->stmt_len;
    next->stmt_str = node->stmt_str;

    // allocate out edges
    n_valid = get_num_valid_sccs2(graph);    
    ca_allocate_out_edges(next, n_valid);

    node->out_edges[e_idx].dst = next;
  }
  else {
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == valid[e_idx]);
  }

  assert (next != NULL);
  printf("selected %s\n", next->name);
  scc_order[(*order_idx)++] = valid[e_idx];
  
  if (filter) {
    free(filter);
    filter = NULL;
  }

  return ca_traverse_statement_order(ant, s_path->next, next, graph, 
				     scc_order, order_idx, vsrc, vdst);
}

/* Ant traverses the graph rooted at node to determine what the fusion / fission
 * partition looks like.
 */
int *traverse_partition_graph(struct ca_ant *ant, struct path *s_path, struct ca_node *node, 
			      struct simple_edges *graph, const int vsrc, const int vdst) {

  printf("traverse partition graph\n");
  assert(node->visited == 0);

  int i;
  struct ca_node *n_node = node_lookup(graph, 1);
  assert (n_node->visited == 0);
  printf("%s -> %s\n", node->name, n_node->name);

  int edge_idx = 0;
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 1);    
    n_node->stmt_len = graph->n_stmts;
    n_node->stmt_str = graph->stmt_str;
    
    node->out_edges[edge_idx].dst = n_node;
  }
  else { // existing edges
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst == n_node) {
	edge_idx = i;
	exists = 1; 
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
      
      node->out_edges[edge_idx].dst = n_node;    
    }
  }  
  
  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;

  node->out_edges[edge_idx].semantic_edge = 1;
  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  //  exit(0);
  // TODO: Use nodes in the graph and n_row to uniquely identify 
  // a root for the next subgraph (if it exists). This will compress 
  // the tree into a graph and ensure that optimization decisions from 
  // one kernel do not affect the decisions made in another
  if (n_node->out_edges == NULL) {
    // allocate the out edges in node  
    const int n_edges = get_num_valid_sccs2(graph);
    ca_allocate_out_edges(n_node, n_edges);
  }

  int *scc_order = (int *)calloc(graph->n_scc, sizeof(int));
  int order_idx = 0;
  printf("begin traversal\n");
  return ca_traverse_statement_order(ant, s_path->next, n_node, graph, 
				     scc_order, &order_idx, vsrc, vdst);
}


int *get_texture_filter(struct ca_ant *ant, int mem) {
  if (!ant->filter_set)
    return NULL;

  int *blacklist = (int *)calloc(2, sizeof(int));

  int lb = ant->filter.attr_lb[texture_start + mem];//ant->opt_vec_ptr];
  int ub = ant->filter.attr_ub[texture_start + mem];//ant->opt_vec_ptr];

  if (lb >= 0) {
    // must be >= lb
    assert (lb == 1);
    blacklist[0] = 1;
  }
  
  if (ub >= 0) {
    // must be <= ub
    assert (ub == 0);
    blacklist[1] = 1;
  }
  return blacklist;
}

int n_mplace_src = 0;
// for each read only array select whether it gets put into texture memory or global memory
char **traverse_texture_graph(struct ca_ant *ant, struct path *s_path, struct ca_node *node, 
			      struct ca_array *read_only, int nread_only, int *n_tex_array) {
  printf("traverse texture\n");
  assert(node->visited == 0);
  int i;
  
  if (!ant->do_reduction) {
    // we expect this to be incremented
    // in the reduction phase
    ++ant->opt_vec_ptr;
  }


  char **tex_array = (char **)calloc(nread_only, sizeof(char *));
  *n_tex_array = 0;

  struct ca_node *n_node = NULL;
  int edge_idx = 0;
  if (node->out_edges == NULL) {
    // add a memplace source node
    ca_allocate_out_edges(node, 1);
    char buf[50];
    snprintf(buf, 50, "mem_place_source:%d\n", n_mplace_src++);
    n_node = create_node(MEM_PLACE_SOURCE, buf, NULL);
    node->out_edges[edge_idx].dst = n_node;
  }
  else { // existing edges
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst->type == MEM_PLACE_SOURCE) {
	edge_idx = i;
	exists = 1; 
	n_node = node->out_edges[i].dst;
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      n_node = create_node(MEM_PLACE_SOURCE, "mem_place_source", NULL);

      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);      
      node->out_edges[edge_idx].dst = n_node;    
    }
  }

  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  s_path = s_path->next;
  node->out_edges[edge_idx].semantic_edge = 1;

  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  for (i = 0; i < nread_only; ++i) {
    if (n_node->out_edges == NULL) {
      ca_allocate_out_edges(n_node, 2);
    }
    
    char *array = strdup(read_only[i].name);

    int *filter = get_texture_filter(ant, i);
    const int e_idx = select_edge(ant, n_node, filter);

    if (filter) {
      free(filter);
      filter = NULL;
    }

    //ant->opt_vec[ant->opt_vec_ptr++] = e_idx;
    ant->opt_vec[texture_start + i] = e_idx;
    ant->opt_vec_ptr++;

    s_path->v = e_idx;
    s_path->next = (struct path *)calloc(1, sizeof(struct path));
    s_path->next->next = NULL;

    assert (e_idx < 2);

    if(e_idx == 1) {
      tex_array[(*n_tex_array)++] = strdup(array);
    }

    assert(n_node->out_edges != NULL);
    struct ca_node *next = NULL;
    if (n_node->out_edges[e_idx].dst == NULL) {
      char node_name[50];
      snprintf(node_name, 50, "%s: %d", array, e_idx);

      // flag 0 -> global mem, 1 -> texture
      int *v = (int *)calloc(1, sizeof(int));
      *v = e_idx;

      next = create_node(MEM_PLACE, node_name, v);
      n_node->out_edges[e_idx].dst = next;
    }
    else {
      assert(n_node != NULL);
      assert(n_node->n_edges == 2);
 
      next = n_node->out_edges[e_idx].dst;
      assert(next->val != NULL);
      assert(*((int *)next->val) == e_idx);
    }
    n_node = next;
    s_path = s_path->next;
  }
  
  return tex_array;
}

int *get_tile_filter(struct ca_ant *ant, int *stmt_id, const int nstmts, 
		     const int coarseness, const int n_edges, const int rem_tile) {  
  int i, j;
  if (!ant->filter_set)
    return NULL;

  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  for (i = 0; i < nstmts; ++i) {
    int lb = ant->filter.attr_lb[stmt_id[i] + tile_start];
    int ub = ant->filter.attr_ub[stmt_id[i] + tile_start];
    
    if (lb >= 0) {
      // coarseness >= lb
      int coarse = 1;
      for (j = n_edges-1; j >= 0; --j) {
	// at n_edges - 1 -> coarsness = 1
	// at n_edges - 2 -> coarsness = 2
	// at n_edges - 3 -> coarsness = 4
	if (coarse < lb) {
	  blacklist[j] = 1;
	}
	coarse *= 2;
      }
    }

    if (ub >= 0) {
      // coarseness <= ub
      int coarse = 1;
      for (j = n_edges-1; j >= 0; --j) {
	// at n_edges - 1 -> coarsness = 1
	// at n_edges - 2 -> coarsness = 2
	// at n_edges - 3 -> coarsness = 4
	if (coarse > ub) {
	  blacklist[j] = 1;
	}
	coarse *= 2;
      }            
    }
  } // end for i
  
  return blacklist;
}

/* If a lower bound exists, make sure the tile is 
 * large enough for the block size to be a multiple 
 * of 32 and still have a granularity > 1.
 */
int *get_tile_size_filter(struct ca_ant *ant, int *stmt_id, 
			  const int nstmts, const int n_edges, 
			  const int rem_dim, const int coarseness) {
  int i, j;
  if (!ant->filter_set)
    return NULL;

  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  for (i = 0; i < nstmts; ++i) {
    int lb = ant->filter.attr_lb[stmt_id[i] + tile_start];
    
    if (lb >= 0) {      
      lb = (lb / coarseness) + 1;
      if (rem_dim == 3) {
	// can have granularity at most 64 with remaining 2 dims
	int idx = 0;
	while (lb > 64) {
	  assert (idx < n_edges);
	  blacklist[idx++] = 1;
	  lb /= 2;
	}
      }
      else if (rem_dim == 2) {
	// can have granularity at most 8 with remaining 1 dims
	int idx = 0;
	while (lb > 8) {
	  assert (idx < n_edges);
	  blacklist[idx++] = 1;
	  lb /= 2;
	}
      }
      else {
	assert (rem_dim == 1);
	// can have granularity at most 8 with remaining 1 dims
	int idx = 0;
	while (lb > 1) {
	  assert (idx < n_edges);
	  blacklist[idx++] = 1;
	  lb /= 2;
	}	 
      }

      // coarseness >= lb
      //for (j = 0; j < lb - 1; ++j) {
      //	assert (j < n_edges);
      //	blacklist[j] = 1;
      //}

    }
  }

  return blacklist;
}


int outer_tile_get_n_edges(int *tiles, const int dim, const int tile_len) {
  int i;

  assert (dim < tile_len);
  assert (tile_len <= 3);

  assert (lg(32) == 5);
  assert (lg(16) == 4);
  assert (lg(2) == 1);

  if (dim == 0 && tile_len == 1) {
    // 32, 64, 128, 256, 512
    return 5;
  }

  if (dim == 0 && tile_len == 2) {
    // 32, 64, 128, 256
    return 4;
  }

  if (dim == 0 && tile_len == 3) {
    // 32, 64, 128
    return 3;
  }

  int size = 1;
  for (i = 0; i < dim; ++i) {
    size *= tiles[i];
  }
  
  int max_block = 1024 / size;

  // 4, 8, 16, 32 for y
  // 2, 4, 8, 16 for z

  if (dim == 1) 
    return lg(max_block) - 1 - (tile_len == 3 ? 1 : 0);

  return lg(max_block);
}

int outer_tile_get_size(const int e_idx, const int dim) {
  if (dim == 0) 
    return 1 << (e_idx + 5);


  if (dim == 1) 
    return 1 << (e_idx + 2);

  return 1 << (e_idx + 1);
}


int **traverse_tile_sizes(struct ca_ant *ant, struct path *s_path, struct ca_node *node, 
			 int *tile_size, int tile_len, char *key, char **stmts, const int nstmts) {
  int i;
  printf("traverse tile sizes\n");
  printf("key = %s\n", key);
  assert(node->visited == 0);
  int **ca_tiles = (int **)calloc(2, sizeof(int *));
  // tile sizes
  ca_tiles[0] = (int *)calloc(tile_len, sizeof(int));

  // block sizes
  ca_tiles[1] = (int *)calloc(tile_len, sizeof(int));

  struct ca_node *n_node = node_lookup_by_key(key);
  int edge_idx = 0;
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 1);
    node->out_edges[edge_idx].dst = n_node;
  }
  else { // existing edges   
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst == n_node) {
	edge_idx = i;
	exists = 1; 
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
    
      node->out_edges[edge_idx].dst = n_node;    
    }
  }

  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL; 
  s_path = s_path->next;
  node->out_edges[edge_idx].semantic_edge = 1;

  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  int *stmt_id = get_stmt_ids(stmts, nstmts); 

  int orig_size = 1;
  int coarseness = 1;
  int block_size = 1;
  for (i = 0; i < tile_len; ++i) {
    if (n_node->out_edges == NULL) {
      int n_edges = outer_tile_get_n_edges(ca_tiles[0], i, tile_len);
      assert (n_edges > 0 && n_edges <= 5);
      ca_allocate_out_edges(n_node, n_edges);
    }

    int *filter = get_tile_size_filter(ant, stmt_id, nstmts, n_node->n_edges, (tile_len - i), coarseness);
    
    int e_idx = select_edge(ant, n_node, filter);

    if (filter) {
      free(filter);
      filter = NULL;
    }

    if (current_iteration < 1)
      n_node->out_edges[e_idx].has_been_traversed = 1;

    s_path->v = e_idx;
    s_path->next = (struct path *)calloc(1, sizeof(struct path));
    s_path->next->next = NULL;
    
    int new_tile = outer_tile_get_size(e_idx, i);
    ca_tiles[0][i] = new_tile;

    orig_size *= tile_size[i];
    coarseness *= new_tile;

    struct ca_node *next = NULL;
    if (n_node->out_edges[e_idx].dst == NULL) {
      char node_name[50];
      snprintf(node_name, 50, "%s %d: %d", "tile", i, new_tile);

      int *v = (int *)calloc(1, sizeof(int));
      *v = new_tile;
      
      next = create_node(CUDA_CONFIG, node_name, v);
      n_node->out_edges[e_idx].dst = next;
    }
    else {
      assert(n_node != NULL);
      next = n_node->out_edges[e_idx].dst;
      assert(*((int *)next->val) == new_tile);
    }
    assert (next != NULL);

    n_node = next;
    s_path = s_path->next;
    
    // set the block size
    if (n_node->out_edges == NULL) {
      // thread block size must be <= tile size
      ca_allocate_out_edges(n_node, e_idx + 1);
    }
    assert (n_node->n_edges == e_idx + 1);

    int *filter_block = get_tile_filter(ant, stmt_id, nstmts, 
				  (coarseness / orig_size),
				  n_node->n_edges, tile_len - i);

    e_idx = select_edge(ant, n_node, filter_block);
    
    if (filter_block) {
      free(filter_block);
      filter_block = NULL;
    }

    s_path->v = e_idx;
    s_path->next = (struct path *)calloc(1, sizeof(struct path));
    s_path->next->next = NULL;
  
    int new_block = outer_tile_get_size(e_idx, i);
    ca_tiles[1][i] = new_block;
    block_size *= new_block;

    next = NULL;
    if (n_node->out_edges[e_idx].dst == NULL) {
      char node_name[50];
      snprintf(node_name, 50, "%s %d: %d", "block", i, new_block);

      int *v = (int *)calloc(1, sizeof(int));
      *v = new_tile;
      
      next = create_node(CUDA_CONFIG, node_name, v);
      n_node->out_edges[e_idx].dst = next;
    }
    else {
      assert(n_node != NULL);
      next = n_node->out_edges[e_idx].dst;
      assert(*((int *)next->val) == new_tile);
    }
    assert (next != NULL);

    n_node = next;
    s_path = s_path->next;

  } // end for i

  coarseness /= block_size;
  for (i = 0; i < nstmts; ++i) {
    ant->opt_vec[tile_start + stmt_id[i]] = coarseness;
  }
  
  free(stmt_id);
  if (ant->sizes == NULL) 
    ant->sizes = int_to_string(ca_tiles[0], tile_len);
  else
    strcat(ant->sizes, int_to_string(ca_tiles[0], tile_len));

  // need to reverse the arrays because of weird quirk in ppcg
  int **tile_blocks = (int **)calloc(2, sizeof(int *));
  tile_blocks[0] = (int *)calloc(tile_len, sizeof(int));
  tile_blocks[1] = (int *)calloc(tile_len, sizeof(int));
  
  for (i = 0; i < tile_len; ++i) {
    tile_blocks[0][i] = ca_tiles[0][tile_len - i - 1];
    tile_blocks[1][i] = ca_tiles[1][tile_len - i - 1];
  }
  
  free(ca_tiles[0]);
  free(ca_tiles[1]);
  free(ca_tiles);

  return tile_blocks;
}

/* Filter out edges corresponding to tile factors < the unroll factor */
int *filter_by_unroll(const int unroll, const int n_edges) {
  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  
  if (unroll >= 8) {
    blacklist[0] = 1; // blacklist 4
  }
  
  if (unroll >= 16) {
    blacklist[1] = 1; // blacklist 8
  }

  if (unroll >= 32) {
    blacklist[2] = 1; // blacklist 16
  }

  if (unroll >= 64) {
    blacklist[3] = 1; // blacklist 32
  }
  return blacklist;
}

int *inner_tile_filter(int *blacklist, struct ca_ant *ant, int *stmt_id, const int nstmts, 
		       const int n_edges, const int size, const int last_tile) {
  int i, j;
  if (!ant->filter_set)
    return blacklist;

  if (!blacklist)
    blacklist = (int *)calloc(n_edges, sizeof(int));

  
  for (i = 0; i < nstmts; ++i) {
    int lb = ant->filter.attr_lb[stmt_id[i] + inner_tile_start];
    int ub = ant->filter.attr_ub[stmt_id[i] + inner_tile_start];
    
    if (lb >= 0) {
      // tile size >= lb
      for (j = n_edges-1; j >= 0; --j) {
	int tile = 1;
	if (j < n_edges - 1)
	  tile = 1 << (j + 2);

	if (tile * size < lb) {
	  if (last_tile) {
	    blacklist[j] = 1;
	  }
	}
      } // end for i
    } // end if lb

    if (ub >= 0) {
      // tile size <= ub
      for (j = 0; j < n_edges; ++j) {
	int tile = 1;
	if (j < n_edges - 1)
	  tile = 1 << (j + 2);

	if (tile * size > ub) {
	  blacklist[j] = 1;
	}
      }
    }
  } // end for i

  return blacklist;
}

int *traverse_inner_tile(struct ca_ant *ant, struct path *s_path, struct ca_node *node, 
			 const int tile_len, int *unroll, char *key, 
			 char **stmts, const int nstmts) {
  int i;
  printf("traverse inner tile\n");
  assert (tile_len > 0);
  int *inner_tiles = (int *)calloc(tile_len, sizeof(int));
 
  struct ca_node *n_node = node_lookup_by_key(key);

  int edge_idx = 0;
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 1);
    node->out_edges[edge_idx].dst = n_node;
  }
  else { // existing edges   
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst == n_node) {
	edge_idx = i;
	exists = 1; 
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
    
      node->out_edges[edge_idx].dst = n_node;    
    }
  }

  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL; 
  s_path = s_path->next;
  node->out_edges[edge_idx].semantic_edge = 1;

  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }


  int *stmt_id = get_stmt_ids(stmts, nstmts); 

  int size = 1;
  for (i = 0; i < tile_len; ++i) {
    if (n_node->out_edges == NULL) {
      // tiles 4, 8, 16, 32, 64, 1
      int n_edges = 5;//(unroll && unroll[i] == 8 ? 4 : 5);
      ca_allocate_out_edges(n_node, n_edges + 1);
    }

    // only select edges that are >= unroll
    int *filter = (unroll ? filter_by_unroll(unroll[i], n_node->n_edges) : NULL);
    filter = inner_tile_filter(filter, ant, stmt_id, nstmts, node->n_edges, 
			       size, (i == tile_len-1));

    const int e_idx = select_edge(ant, n_node, filter);

    if (filter) {
      free(filter); 
      filter = NULL;
    }

    s_path->v = e_idx;
    s_path->next = (struct path *)calloc(1, sizeof(struct path));
    s_path->next->next = NULL;
    
    int tile = 1;
    if (e_idx < n_node->n_edges - 1)
      tile = 1 << (e_idx + 2);

    assert (tile == 4 || tile == 8 || tile == 16 || tile == 32 || tile == 64 || tile == 1);
    
    inner_tiles[i] = tile;
    size *= tile;

    struct ca_node *next = NULL;
    if (n_node->out_edges[e_idx].dst == NULL) {
      char name[50];
      snprintf(name, 50, "inner tile %d", tile);

      int *v = (int *)calloc(1, sizeof(int));
      *v = tile;
      next = create_node(CUDA_CONFIG, name, v);
      n_node->out_edges[e_idx].dst = next;
    }
    else {
      next = n_node->out_edges[e_idx].dst;
      assert(next);
      assert (*((int *)next->val) == tile);
    }
    assert(next);
    n_node = next;
    s_path = s_path->next;
  }
  
  // TODO: ADD ANT OPT_VEC
  for (i = 0; i < nstmts; ++i) {
    ant->opt_vec[inner_tile_start + stmt_id[i]] = size;
  }
  free(stmt_id);

  printf("inner tile = %s\n", int_to_string(inner_tiles, tile_len));
  return inner_tiles;
}


int *get_unroll_filter(struct ca_ant *ant, int *stmt_id, const int nstmts, const int n_edges) {
  int i, j;
  if (!ant->filter_set)
    return NULL;

  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  for (i = 0; i < nstmts; ++i) {
    int lb = ant->filter.attr_lb[stmt_id[i] + unroll_start];
    int ub = ant->filter.attr_ub[stmt_id[i] + unroll_start];
    
    if (lb >= 0) {
      // must be >= lb
      for (j = 0; j < lb; ++j) {
	blacklist[j] = 1;
      }
    }

    if (ub >= 0) {
      // must be <= ub
      for (j = ub + 1; j < n_edges; ++j) {
	blacklist[j] = 1;
      }
    }    
  } // end for i

  return blacklist;  
}

int n_valid_unroll(const int nregs, const int max_regs) {
  if (8 * nregs <= max_regs) 
    return 4;
  else if (4 * nregs <= max_regs)
    return 3;
  else if (2 * nregs <= max_regs)
    return 2;
  
  return 1;
}

// get the upper bound on the number of registers based on 
// the min upper bound of all statements in stmt_id
int get_unroll_ub_regs(struct ca_ant *ant, int *stmt_id, const int nstmts) {
  int i;
  if (!ant->filter_set)
    return MAX_REGS;
  
  int min_max = MAX_REGS;
  for (i = 0; i < nstmts; ++i) {
    int ub = ant->filter.attr_ub[stmt_id[i] + unroll_start];
    if (ub >= 0) {
      min_max = (ub < min_max ? ub : min_max);
    }
  }
  
  return min_max;
}

int *get_unroll_lb_filter(struct ca_ant *ant, int *stmt_id, const int nstmts, 
			  const int n_edges, const int nregs) {
  int i;
  if (!ant->filter_set)
    return NULL;

  assert (nregs > 0);
  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  int max_min = 0;
  for (i = 0; i < nstmts; ++i) {
    int lb = ant->filter.attr_lb[stmt_id[i] + unroll_start];
    
    if (lb >= 0) {
      max_min = (lb > max_min ? lb : max_min);
    }
  }
  
  int regs = nregs;
  for (i = 0; i < n_edges; ++i) {
    if (regs < max_min) {
      blacklist[i] = 1;
    }
    else {
      return blacklist;
    }
    regs *= 2;    
  }

  // if all are unblocked, then unblock the last one
  blacklist[n_edges - 1] = 0;
  return blacklist;
}

int *traverse_unroll_factors(struct ca_ant *ant, struct path *s_path, struct ca_node *node, 
			     const int nregs, const int n_unroll_factors, 
			     char **stmts, const int nstmts) {
  printf("traverse unroll factors\n");
  int i;

  // 0 -> no unroll
  // 1 -> unroll 2
  // 2 -> unroll 4
  // 3 -> unroll 8
  
  int *stmt_id = get_stmt_ids(stmts, nstmts); 
  const int max_regs = get_unroll_ub_regs(ant, stmt_id, nstmts);

  int *unroll = (int *)calloc(n_unroll_factors, sizeof(int));
  int total_regs = nregs;
  for (i = 0; i < n_unroll_factors; ++i) {
    printf("total_regs = %d\n", total_regs);
    if (node->out_edges == NULL) {
      ca_allocate_out_edges(node, n_valid_unroll(total_regs, max_regs));
    }

    int *filter = get_unroll_lb_filter(ant, stmt_id, nstmts, 
				       node->n_edges, total_regs);
        
    const int e_idx = select_edge(ant, node, filter);
    
    if (filter) {
      free(filter);
      filter = NULL;
    }

    s_path->v = e_idx;
    s_path->next = (struct path *)calloc(1, sizeof(struct path));
    s_path->next->next = NULL;
    s_path = s_path->next;

    int unroll_factor = 1 << e_idx;
    printf("unroll_factor = %d\n", unroll_factor);
    assert (node->out_edges != NULL);
    struct ca_node *next = NULL;
    if (node->out_edges[e_idx].dst == NULL) {
      printf("here1\n");
      char name[50];
      snprintf(name, 50, "unroll %d", unroll_factor);
      
      int *v = (int *)calloc(1, sizeof(int));
      *v = unroll_factor;
      
      next = create_node(UNROLL, name, v);
      node->out_edges[e_idx].dst = next;
    }
    else {
      printf("here2\n");
      next = node->out_edges[e_idx].dst;
      printf("here3\n");
      assert(next);
      assert (*((int *)next->val) == unroll_factor);
    }
    assert(next);

    node = next;

    unroll[i] = unroll_factor;
    total_regs *= unroll_factor;
  } // end for i
  
  for (i = 0; i < nstmts; ++i) {
    ant->opt_vec[unroll_start + stmt_id[i]] = total_regs;
  }
  free(stmt_id);
  
  if (ant->unroll_factors == NULL) {
    ant->unroll_factors = (char *)calloc(1000, sizeof(char));
    snprintf(ant->unroll_factors, 1000, "%s,", int_to_string(unroll, n_unroll_factors)); 
  }
  else {
    char buf[100];
    snprintf(buf, 100, "%s, ", int_to_string(unroll, n_unroll_factors)); 
    strcat(ant->unroll_factors, buf);
  }
  return unroll;
}

int traverse_unroll_factors2(struct ca_ant *ant, struct path *s_path, struct ca_node *node,
			    char **stmts, const int nstmts) {
  int i;
  printf("traverse unroll factors\n");

  struct ca_node *n_node = node;
  int edge_idx = 0;
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 1);
    n_node = create_node(UNROLL_SOURCE, "unroll_source", NULL);
    node->out_edges[edge_idx].dst = n_node;
  }
  else {
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst->type == UNROLL_SOURCE) {
	edge_idx = i;
	exists = 1; 
	n_node = node->out_edges[edge_idx].dst;
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
      n_node = create_node(UNROLL_SOURCE, "unroll_source", NULL);
      node->out_edges[edge_idx].dst = n_node;    
    }
  }
  
  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL; 
  s_path = s_path->next;
  node->out_edges[edge_idx].semantic_edge = 1;
    
  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  if (n_node->out_edges == NULL) {
    // 0 -> no unroll   
    // 1 -> unroll factor 4
    // 2 -> unroll factor 8
    // 3 -> full unroll
    ca_allocate_out_edges(n_node, 4);
  }

  assert(n_node->n_edges == 4);
  
  int *stmt_id = get_stmt_ids(stmts, nstmts); 
  int *filter = get_unroll_filter(ant, stmt_id, nstmts, n_node->n_edges);

  const int e_idx = (current_iteration < 1 ? 0 : select_edge(ant, n_node, filter));
  if (filter) {
    free(filter);
    filter = NULL;
  }

  if (current_iteration < 1)
    n_node->out_edges[e_idx].has_been_traversed = 1;

  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  
  for (i = 0; i < nstmts; ++i) {
    ant->opt_vec[unroll_start + stmt_id[i]] = e_idx;
  }
  free(stmt_id);

  int unroll = 0;
  if (e_idx == 0) 
    unroll = 0;
  else if (e_idx == 1)
    unroll = 4;
  else if (e_idx == 2)
    unroll = 8;
  else if (e_idx == 3)
    unroll = -1;

  struct ca_node *next = NULL;
  if (n_node->out_edges[e_idx].dst == NULL) {
    char node_name[50];
    snprintf(node_name, 50, "unroll %d", unroll);

    int *v = (int *)calloc(1, sizeof(int));
    *v = unroll;
      
    next = create_node(UNROLL, node_name, v);
    n_node->out_edges[e_idx].dst = next;
  }
  else {
    assert(n_node != NULL);
    assert(n_node->n_edges == 4);
    
    next = n_node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == unroll);
  }

  if (ant->unroll_factors == NULL) {
    ant->unroll_factors = (char *)calloc(1000, sizeof(char));
    snprintf(ant->unroll_factors, 1000, "[%d,", unroll); 
  }
  else {
    char buf[50];
    snprintf(buf, 50, "%d, ", unroll); 
    strcat(ant->unroll_factors, buf);
  }
  return unroll;
}


int traverse_atomic_store(struct ca_ant *ant, struct path *s_path, 
			  struct ca_node *node, char *key) {
  int i;
  printf("traverse atomic store\n");
  struct ca_node *n_node = node_lookup_by_key(key);
  int edge_idx = 0;
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 1);
    node->out_edges[edge_idx].dst = n_node;
  }
  else {
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst == n_node) {
	edge_idx = i;
	exists = 1; 
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
    
      node->out_edges[edge_idx].dst = n_node;    
    }
  }
  
  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL; 
  s_path = s_path->next;
  node->out_edges[edge_idx].semantic_edge = 1;

  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  if (n_node->out_edges == NULL) {
    // 0 -> no atomic
    // 1 -> atomic
    ca_allocate_out_edges(n_node, 2);
  }
  assert (n_node->n_edges == 2);
 
  const int e_idx = select_edge(ant, n_node, NULL);
  
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  s_path = s_path->next;

  struct ca_node *next = NULL;
  if (n_node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx ? "use atomic store" : "no atomic store");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;
      
    next = create_node(ATOMIC, name, v);
    n_node->out_edges[e_idx].dst = next;  
  }
  else {
    assert(n_node != NULL);
    assert(n_node->n_edges == 2);

    next = n_node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);  
  }
  assert (next);

  /*if (ant->unroll_factors == NULL) {
    ant->unroll_factors = (char *)calloc(1000, sizeof(char));
    snprintf(ant->unroll_factors, 1000, "[%d,", unroll); 
  }
  else {
    char buf[50];
    snprintf(buf, 50, "%d, ", unroll); 
    strcat(ant->unroll_factors, buf);
    }*/
  printf("done\n");
    
  return e_idx;
}

int traverse_cache_config(struct ca_ant *ant, struct path *s_path, 
			  struct ca_node *node, const int is_shared) {
  int i;
  printf("traverse cache config\n");
  assert(node->type == SHARED);
  printf("starting node = %s\n", node->name);
  assert (is_shared <= 1);
  if (node->out_edges == NULL) {
    if (is_shared)
      ca_allocate_out_edges(node, 3);
    else
      ca_allocate_out_edges(node, 2);
  }

  if (is_shared) {
    printf("nedges = %d\n", node->n_edges);
    assert(node->n_edges == 3);
  }
  else {
    printf("nedges = %d\n", node->n_edges);
    assert(node->n_edges == 2);
  }
  const int e_idx = select_edge(ant, node, NULL);
  
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  s_path = s_path->next;

  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char *name;
    if (is_shared == 0) {
      name = (e_idx == 0 ? "cache config: no preference" : "cache config: L1 preference");
    }
    else {
      name = (e_idx == 0 ? "cache config: shared preference" :
	      (e_idx == 1 ? "cache config: no preference" : "cache config: L1 preference"));
    }
    
    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;

    next = create_node(CACHE, name, v);
    node->out_edges[e_idx].dst = next;
  }
  else{
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);
  }

  // 0 -> prefer shared
  // 1 -> prefer none
  // 2 -> perfer L1
  printf("end of traverse cache\n");
  return (is_shared == 0 ? e_idx + 1 : e_idx);
}

int traverse_shared_memory(struct ca_ant *ant, struct path *s_path, 
			   struct ca_node *node, char *key) {
  printf("traverse shared memory\n");
  int i;
  struct ca_node *n_node = node_lookup_by_key(key);
  int edge_idx = 0;
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 1);
    node->out_edges[edge_idx].dst = n_node;
  }
  else {
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst == n_node) {
	edge_idx = i;
	exists = 1; 
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
    
      node->out_edges[edge_idx].dst = n_node;    
    }
  } // end else


  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL; 
  s_path = s_path->next;
  node->out_edges[edge_idx].semantic_edge = 1;

  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  if (n_node->out_edges == NULL) {
    ca_allocate_out_edges(n_node, 2);
  }
  assert (n_node->n_edges == 2);
  
  const int e_idx = select_edge(ant, n_node, NULL);
  
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  s_path = s_path->next;
  
  struct ca_node *next = NULL;
  if (n_node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx == 0 ? "no shared memory" : "use shared memory");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;
      
    next = create_node(SHARED, name, v);
    n_node->out_edges[e_idx].dst = next;
  }
  else {    
    next = n_node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);
  }
  assert (next);

  return e_idx;
}

isl_vec *ca_traverse_schedule(struct ca_ant *ant, struct path *s_path, 
			      struct ca_node *node, isl_mat *schedules) {
  int i;

  assert (node->out_edges != NULL);
  const int e_idx = select_edge(ant, node, NULL);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;

  isl_vec *sol = isl_mat_get_row(schedules, e_idx);
  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char buf[100];
    strcpy(buf, "");
    for (i = 0; i < isl_vec_size(sol); ++i) {
      char sv[5];
      isl_val *val = isl_vec_get_element_val(sol, i);
      snprintf(sv, 5, "%s,", isl_val_to_str(val));
      strcat(buf, sv);
    }

    next = create_node(SCHED, buf, isl_vec_copy(sol));
    next->stmt_len = node->stmt_len;
    next->stmt_str = node->stmt_str;
    node->out_edges[e_idx].dst = next;    
  }
  else {
    next = node->out_edges[e_idx].dst;    
  }
  printf("selected  ");
  isl_vec_dump(sol);

  return sol;
}

int is_skew(isl_vec *s) {
  int i;
  
  int non_zeros = 0;
  for (i = 0; i < isl_vec_size(s); ++i) {
    isl_val *val = isl_vec_get_element_val(s, i);
    if (!isl_val_is_zero(val)) {
      ++non_zeros;
      if (non_zeros > 1) {
	return 1;
      }
    }
  }
  
  return 0;
}

int is_interchange(isl_vec *s) {
  return !is_skew(s);
}

int contains_skew(isl_mat *schedules) {
  int i;
  const int rows = isl_mat_rows(schedules);
  for (i = 0; i < rows; ++i) {
    isl_vec *s = isl_mat_get_row(schedules, i);
    if (is_skew(s)) {
      return 1;
    }
  }
  
  return 0;
}

int contains_interchange(isl_mat *schedules) {
  int i;
  const int rows = isl_mat_rows(schedules);
  for (i = 0; i < rows; ++i) {
    isl_vec *s = isl_mat_get_row(schedules, i);
    if (is_interchange(s)) {
      return 1;
    }
  }
  
  return 0;
}


/* Get a marix of valid schedule rows from 'schedules' that are of
 * a specific type defined by 'skew'.
 * schedules matrix of valid schedule rows
 * skew if 1 then only select rows corresponding to a skew. Otherwise, 
 * select rows corresponding to interchange
 * return a matrix of rows from 'schedules' that have the appropriate 
 * schedule type
 */
isl_mat *get_schedules_with_type(isl_mat *schedules, int skew) {
  int i;
  isl_mat *valid = NULL;
  const int rows = isl_mat_rows(schedules);
  for (i = 0; i < rows; ++i) {
    isl_vec *s = isl_mat_get_row(schedules, i);
    int add = (skew ? is_skew(s) : is_interchange(s));
    if (add) {
      if (!valid) {
	valid = isl_mat_from_row_vec(isl_vec_copy(s));
      }
      else {
	valid = isl_mat_vec_concat(valid, isl_vec_copy(s));
      }
    }
  }

  return valid;
}


int *get_schedule_type_filter(struct ca_ant *ant, const int n_edges, const int stmt_id) {
  int i;
  if (!ant->first_partition || !ant->filter_set)
    return NULL;

  if (n_edges < 2)
    return NULL;

  int *blacklist = (int *)calloc(n_edges, sizeof(int));
  int lb = ant->filter.attr_lb[stmt_id + skewing_start];
  int ub = ant->filter.attr_ub[stmt_id + skewing_start];

  if (lb >= 0) {
    assert (lb == 1);
    blacklist[0] = 1;
  }

  if (ub >= 0) {    
    assert (ub == 0);
    blacklist[1] = 1;
  }

  return blacklist;
}

isl_vec *ca_traverse_schedule_type(struct ca_ant *ant, struct path *s_path, 
				   struct ca_node *node, isl_mat *schedules,
				   struct simple_edges *graph, const int stmt_id) {
  int i;
  
  assert (node->out_edges != NULL);
  //assert(node->n_edges == 2);

  int *filter = get_schedule_type_filter(ant, node->n_edges, stmt_id);

  const int e_idx = select_edge(ant, node, filter);

  if (filter) {
    free(filter);
    filter = NULL;
  }

  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;

  isl_mat *sched_type = (node->n_edges > 1 ? 
			 get_schedules_with_type(schedules, e_idx) :
			 schedules);
  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx == 0 ? "interchange" : "skew");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;

    next = create_node(SCHED_TYPE, name, v);
    next->stmt_len = node->stmt_len;
    next->stmt_str = node->stmt_str;
    
    node->out_edges[e_idx].dst = next;

    const int n_edges = isl_mat_rows(sched_type);
    ca_allocate_out_edges(next, n_edges);
  }
  else {
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);
    assert(next->n_edges == isl_mat_rows(sched_type));
  }
  
  //if (ant->first_partition) {
    
    printf("nedges = %d\n", node->n_edges);
    printf("e_idx = %d\n", e_idx);
    assert (e_idx < 2);

    //printf("edges = %d\n", node->n_edges);
    //printf("%s\n", int_to_string(ant->opt_vec, num_regression_vec));
    ant->opt_vec[skewing_start + stmt_id] = e_idx;
    ant->first_partition = 0;
    printf("sched. type %s\n", int_to_string(ant->opt_vec, num_regression_vec));
    //}

  return ca_traverse_schedule(ant, s_path->next, next, sched_type);
}

isl_vec *traverse_schedule_graph(struct ca_ant *ant, struct path *s_path, 
				 struct ca_node *node, struct simple_edges *graph,
				 isl_mat *schedules, const int stmt_id) {
  int i;
  printf("traverse schedule graph\n");
  assert(node->visited == 0);
  
  struct ca_node *n_node = node_lookup(graph, 0);
  assert(n_node->visited == 0);
  printf("%s -> %s\n", node->name, n_node->name);  

  int edge_idx = 0;
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 1);    
    n_node->stmt_len = graph->n_stmts;
    n_node->stmt_str = graph->stmt_str;

    node->out_edges[edge_idx].dst = n_node;
  }
  else { // existing edges
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst == n_node) {
	edge_idx = i;
	exists = 1; 
	break;
      }
    }
    
    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
    
      node->out_edges[edge_idx].dst = n_node;    
    }
  }  

  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  node->out_edges[edge_idx].semantic_edge = 1;
  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  if (n_node->out_edges == NULL) {
    // for now just select one of the rows
    // from the 'schedules' matrix. 
    // TODO: Add additional layer distinguishing 
    // interchange from skewing
    
    int n_edges = 0;//isl_mat_rows(schedules);
    if (contains_skew(schedules)) {
      ++n_edges;
    }

    if (contains_interchange(schedules)) {
      ++n_edges;
    }

    assert(n_edges > 0);
    ca_allocate_out_edges(n_node, n_edges);
  }
  return ca_traverse_schedule_type(ant, s_path->next, n_node, schedules, graph, stmt_id);
  //  return ca_traverse_schedule(s_path->next, n_node, schedules);
}

int *get_reduction_filter(struct ca_ant *ant) {

  if (!ant->filter_set)
    return NULL;

  int *blacklist = (int *)calloc(2, sizeof(int));

  int ub = ant->filter.attr_ub[reduction_start];
  
  // turns off reduction
  if (ub >= 0) {
    // must be <= ub
    assert (ub == 0);
    blacklist[1] = 1;
  }
  return blacklist;
}

int ca_traverse_reduction(struct ca_ant *ant, struct path *s_path, struct ca_node *node) {
  int i;
  
  assert(node->out_edges != NULL);
  assert(node->n_edges == 2);

  int *filter = get_reduction_filter(ant);

  const int e_idx = select_edge(ant, node, filter);

  if (filter) {
    free(filter);
    filter = NULL;
  }
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;

  if (!ant->do_reduction && e_idx == 1) {
    ant->do_reduction = 1;
    ant->opt_vec[reduction_start] = 1;
    ant->opt_vec_ptr++;
  }

  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx == 0 ? "reduction disabled" : "reduction enabled");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;

    next = create_node(TOGGLE_RED, name, v);
    next->stmt_len = node->stmt_len;
    next->stmt_str = node->stmt_str;

    next->in_kernel = 1;//node->in_kernel;
    node->out_edges[e_idx].dst = next;   
  }
  else {
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);  
  }

  return e_idx;
}

int traverse_reduction_graph(struct ca_ant *ant, struct path *s_path, 
			     struct ca_node *node, struct simple_edges *graph) {
  int i;
  printf("traverse reduction graph\n" );
  assert(node->visited == 0);
  
  struct ca_node *n_node = node_lookup(graph, 2);
  n_node->in_kernel = node->in_kernel;
  assert(n_node->visited == 0);
    
  int edge_idx = 0;
  if (node->out_edges == NULL) {
    n_node->stmt_len = graph->n_stmts;
    n_node->stmt_str = graph->stmt_str;

    ca_allocate_out_edges(node, 1);    
    node->out_edges[edge_idx].dst = n_node;
  }
  else { // existing edges  
    int exists = 0;
    // check if n_node is already an edge
    for (i = 0; i < node->n_edges; ++i) {
      if (node->out_edges[i].dst == n_node) {
	edge_idx = i;
	exists = 1; 
	break;
      }
    }

    if (!exists) {    
      // add edge from node to n_node
      edge_idx = node->n_edges;
      ca_realloc_edges(node, node->n_edges + 1);
      
      node->out_edges[edge_idx].dst = n_node;    
    }
  }

  node->out_edges[edge_idx].has_been_traversed = 1;
  s_path->v = edge_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  node->out_edges[edge_idx].semantic_edge = 1;
  if(best_first_search) {
    if (current_depth+1 == beam_depth && node == beam_node) {
      semantic_edge = edge_idx;
    }
  }

  if (n_node->out_edges == NULL) {
    // one for reduction enabled and one for disabled
    ca_allocate_out_edges(n_node, 2);
  }
  
  return ca_traverse_reduction(ant, s_path->next, n_node);
}

// partition graph that is fully a tree
// useful to test how space the graph is saving when we 
// use the lookup table method
int *traverse_partition_graph2(struct ca_ant *ant, struct path *s_path, struct ca_node *node, 
			      struct simple_edges *graph) {

  assert(node->visited == 0);

  int i;
  if (node->out_edges == NULL) {
    // allocate the out edges in node  
    const int n_edges = get_num_valid_sccs2(graph);
    ca_allocate_out_edges(node, n_edges);
  }

  int *scc_order = (int *)calloc(graph->n_scc, sizeof(int));
  int order_idx = 0;
  printf("begin traversal\n");
  return NULL;//ca_traverse_statement_order(ant, s_path, node, graph, scc_order, &order_idx);
}

void select_outer_coin(struct ca_ant *ant, struct ca_node *node, struct path *s_path) {
  assert(node->n_edges == 2);
  
  const int e_idx = select_edge(ant, node, NULL);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  
  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx == 0 ? "no outer coincidence" : "outer coincidence");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;

    next = create_node(CUDA_CONFIG, name, v);
    node->out_edges[e_idx].dst = next;   
  }
  else {
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);   
  }

  ant->compiler_options[CO_OUTER_COIN] = e_idx;
}

void select_max_band_depth(struct ca_ant *ant, struct ca_node *node, struct path *s_path) {
  assert(node->n_edges == 2);
  
  const int e_idx = select_edge(ant, node, NULL);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  
  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx == 0 ? "no max band depth" : "max band depth");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;

    next = create_node(CUDA_CONFIG, name, v);
    node->out_edges[e_idx].dst = next;   
  }
  else {
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);   
  }

  ant->compiler_options[CO_MAX_DEPTH] = e_idx;

  if (next->out_edges == NULL) {
    ca_allocate_out_edges(next, 2);
  }

  select_outer_coin(ant, next, s_path->next);
}

void select_serialize_sccs(struct ca_ant *ant, struct ca_node *node, struct path *s_path) {
  assert(node->n_edges == 2);

  const int e_idx = select_edge(ant, node, NULL);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
  
  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx == 0 ? "no serialize sccs" : "serialize sccs");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;

    next = create_node(CUDA_CONFIG, name, v);
    node->out_edges[e_idx].dst = next;   
  }
  else {
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);   
  }

  ant->compiler_options[CO_SERIALIZE] = e_idx;

  if (next->out_edges == NULL) {
    ca_allocate_out_edges(next, 2);
  }

  if (e_idx == 0)
    select_max_band_depth(ant, next, s_path->next);
  else 
    select_outer_coin(ant, next, s_path->next);
}

void select_alg(struct ca_ant *ant, struct ca_node *node, struct path *s_path) {
  assert(node->n_edges == 2);
  
  const int e_idx = select_edge(ant, node, NULL);
  s_path->v = e_idx;
  s_path->next = (struct path *)calloc(1, sizeof(struct path));
  s_path->next->next = NULL;
 
  struct ca_node *next = NULL;
  if (node->out_edges[e_idx].dst == NULL) {
    char *name = (e_idx == 0 ? "ISL" : "FEAUTRIER");

    int *v = (int *)calloc(1, sizeof(int));
    *v = e_idx;

    next = create_node(CUDA_CONFIG, name, v);
    node->out_edges[e_idx].dst = next;   
  }
  else {
    next = node->out_edges[e_idx].dst;
    assert(*((int *)next->val) == e_idx);   
  }

  ant->compiler_options[CO_ALG] = e_idx;

  if (next->out_edges == NULL) {
    ca_allocate_out_edges(next, 2);
  }

  select_serialize_sccs(ant, next, s_path->next);
}

void traverse_compiler_options(struct ca_ant *ant, struct ca_node *node, struct path *s_path) {
  
  if (node->out_edges == NULL) {
    ca_allocate_out_edges(node, 2);
  }
  
  // select algorithm
  select_alg(ant, node, s_path);
}

void ca_traverse(struct ca_ant *ant, struct coding_ants *ca) {
  int i, j;
  if (ca->wrap->optimization > 0)
    return;

  traverse_compiler_options(ant, ca->wrap->node, ca->wrap->s_path);
  if (0) { // fix for correlation lexmin bug
    while (ant->compiler_options[CO_ALG] == 1 && ant->compiler_options[CO_SERIALIZE] == 0 && ant->compiler_options[CO_MAX_DEPTH] == 0 && ant->compiler_options[CO_OUTER_COIN] == 0) {
      printf("re traverse\n");
      traverse_compiler_options(ant, ca->wrap->node, ca->wrap->s_path);
    }

    while (ant->compiler_options[CO_ALG] == 0 && ant->compiler_options[CO_SERIALIZE] == 0 && ant->compiler_options[CO_MAX_DEPTH] == 1 && ant->compiler_options[CO_OUTER_COIN] == 0) {
      printf("re traverse\n");
      traverse_compiler_options(ant, ca->wrap->node, ca->wrap->s_path);
    }

    while (ant->compiler_options[CO_ALG] == 1 && ant->compiler_options[CO_SERIALIZE] == 0 && ant->compiler_options[CO_MAX_DEPTH] == 1 && ant->compiler_options[CO_OUTER_COIN] == 0) {
      printf("re traverse\n");
      traverse_compiler_options(ant, ca->wrap->node, ca->wrap->s_path);
    }

    if (ant->compiler_options[CO_ALG] == 1 && ant->compiler_options[CO_SERIALIZE] == 0 && ant->compiler_options[CO_MAX_DEPTH] == 0 && ant->compiler_options[CO_OUTER_COIN] == 0) {
      ant->compiler_options[CO_ALG]=0;
    }

    // for covariance
    if (ant->compiler_options[CO_ALG] == 0 && ant->compiler_options[CO_SERIALIZE] == 0 && ant->compiler_options[CO_MAX_DEPTH] == 1 && ant->compiler_options[CO_OUTER_COIN] == 0) {
      ant->compiler_options[CO_MAX_DEPTH]=0;
    }

    while (ant->compiler_options[CO_ALG] == 0 && ant->compiler_options[CO_SERIALIZE] == 0 && ant->compiler_options[CO_MAX_DEPTH] == 0 && ant->compiler_options[CO_OUTER_COIN] == 0) {
      printf("re traverse\n");
      traverse_compiler_options(ant, ca->wrap->node, ca->wrap->s_path);
    }
  }
  // update wrap node to point to the current leaf
  struct ca_node *node = ca->wrap->node;
  struct path *s_path = ca->wrap->s_path;
  
  while(s_path->next != NULL) {
    assert(node != NULL);
    assert(node->out_edges != NULL);
    assert(node->n_edges > 0);
    
    assert(s_path->v < node->n_edges);
    node = node->out_edges[s_path->v].dst;
    s_path = s_path->next;
    
    assert(s_path != NULL);
  }
  assert(node != NULL);

  ca->wrap->s_path = s_path;
  ca->wrap->node = node;   
}

void decay(struct ca_node *node) {
  int i;
  
  if (node == NULL || node->out_edges == NULL || node->visited) {
    // base case
    return;
  }
  
  node->visited = 1;
  for (i = 0; i < node->n_edges; ++i) {
    // (1) decay the pheremone
    double tau = (1 - RHO) * node->out_edges[i].tau;
    node->out_edges[i].tau = (tau < MIN_TAU ? MIN_TAU : tau);

    // dfs
    decay(node->out_edges[i].dst);
  }
  
  //node->visited = 0;
}

double F(double score) {
  return 1.0 / score;
}

double F_cycles(unsigned long score) {
  // number of seconds
  double s = (double) score / 1e9;
  
  return 1.0 / s;
}


void deposit_best(struct ca_node *node, struct path *best_path, double score) {
  int i;
  assert (node != NULL);
  
  if (node->out_edges == NULL || best_path == NULL) { // base case    
    return;
  }   
  
  assert(best_path->v < node->n_edges); 
  assert (node->out_edges[best_path->v].dst != NULL);
  struct ca_node *next = node->out_edges[best_path->v].dst;
  
  double delta = (score > 10000 ? 0 : RHO * 1.0);
  node->out_edges[best_path->v].tau += delta;
  node->out_edges[best_path->v].tau = (node->out_edges[best_path->v].tau > MAX_TAU ? 
				       MAX_TAU : node->out_edges[best_path->v].tau);
  deposit_best(next, best_path->next, score);
}

void deposit_sched2(struct ca_ant *ant, struct ca_node *node, struct path *path_list, 
		    struct path *best_path,
		    double score, struct coding_ants *ca) {
  int i;
  assert (node != NULL);
  if (node->out_edges == NULL || path_list == NULL) { // base case    
    return;
  }   
  assert(path_list != NULL);
  assert(path_list->v < node->n_edges); 
  assert(path_list->v >= 0);
  printf("%s, %d\n", node->name, path_list->v);
  printf("is next null = %d\n", (path_list->next == NULL));
  assert (node->out_edges[path_list->v].dst != NULL);

  struct ca_node *next = node->out_edges[path_list->v].dst;
  char *key = hash_kern(ant, next);
  int kernel = get_node_kernel(ant, next); 
  printf("kern = %d, key = %s\n\n", kernel, key);
  if (next->in_kernel && key && ant->kernel_cycles[kernel] > 0) {
    printf("path list %d\n", path_list->v);
     struct kern_time_tuple *best_kern = kern_table_lookup(key);

    assert (best_kern != NULL);

    //printf("%lu, %lu\n", best_kern->time, next->kernel_score);
    assert (next->kernel_score >= best_kern->time);
    //if (next->kernel_score == best_kern->time) {
    if (((double) next->kernel_score / (double)best_kern->time) >= 0.95) {
    //if (best_kern->time > 0) {
      //double delta = (score > 10000 ? 0 : CA_Q * ((double)best_kern->time / 
      //					  (double)ant->kernel_cycles[kernel]));
      double delta = (score > 10000 ? 0 : 0.5 * RHO * 1.0);
      assert (delta == delta); // check for nan
      node->out_edges[path_list->v].tau += delta; 
      node->out_edges[path_list->v].tau = (node->out_edges[path_list->v].tau > MAX_TAU ? MAX_TAU :
				       node->out_edges[path_list->v].tau);

    }
  }
  else {
    //double delta = (score > 10000 ? 0 : RHO * F(score));
    //node->out_Edges[path_list->v].tau += delta;
  }
  
  deposit_sched2(ant, next, path_list->next, best_path, score, ca);
}

void deposit_kernels(struct ca_node *node) {
  struct kern_time_tuple *trav = kernel_best_time_table;
  while(trav != NULL) {
    // push ant changes
    /* if (!trav->push) {
      printf("d1\n");
      printf("texs = %s\n", trav->ant->texs[0]);
      trav->ant = copy_ant(*(trav->ant));
      printf("d2\n");
      exit(0);
      trav->push = 1;
      }*/
  if (trav->ant) {
    deposit_sched2(trav->ant, node, trav->ant->s_path, trav->ant->s_path, trav->ant->score, NULL);
  }
  trav = trav->next;
  }
}

void deposit_sched2_back2(struct ca_ant *ant, struct ca_node *node, struct path *path_list, 
		    struct path *best_path,
		    double score, struct coding_ants *ca) {
  int i;
  printf("inkernel = %s, kernel = %d, deposits %s\n", (node->in_kernel ? "true" : "false"), 
	 get_node_kernel(ant, node), node->name);
  assert (node != NULL);
  assert (ca != NULL);

  if (node->out_edges == NULL || path_list == NULL) { // base case    
    return;
  }   
  
  assert(path_list != NULL);
  
  double best_score = ca->best_current_time;
  
  assert(path_list->v < node->n_edges); 
  assert (node->out_edges[path_list->v].dst != NULL);
  
  struct ca_node *next = node->out_edges[path_list->v].dst;
  char *key = hash_kern(ant, next);
  int kernel = get_node_kernel(ant, next);    

  if (next->in_kernel && key && ant->kernel_cycles[kernel] > 0) {
    assert (key != NULL);    
    struct kern_time_tuple *best_kern = kern_table_lookup(key);
    
    printf("%lu, %lu\n", best_kern->time, next->kernel_score);
    assert (next->kernel_score >= best_kern->time);

    //if (next->kernel_score == best_kern->time) {
    if (((double) next->kernel_score / (double)best_kern->time) >= 0.95) {
    //if (best_kern->time > 0) {
      double delta = (score > 10000 ? 0 : CA_Q * ((double)best_kern->time / 
						  (double)ant->kernel_cycles[kernel]));
      assert (delta == delta); // check for nan
      node->out_edges[path_list->v].tau += delta;    
    }
  }
  else {
    //if (abs(next->best_score - best_score) <= 0.1) {
    if ((best_score / next->best_score) >= 0.95) {
    //if (1) {
      // this ant is on the best path
      double delta = (score > 10000 ? 0 : CA_Q * (best_score / score));
      assert (delta == delta); // check for nan
      node->out_edges[path_list->v].tau += delta;    
    }
  }
  
  deposit_sched2(ant, next, path_list->next, best_path, score, ca);
}

void deposit_sched2_back(struct ca_ant *ant, struct ca_node *node, struct path *path_list, 
		    struct path *best_path,
		    double score, struct coding_ants *ca) {
  int i;
  printf("inkernel = %s, kernel = %d, deposits %s\n", (node->in_kernel ? "true" : "false"), 
	 node->kernel_id, node->name);
  assert (node != NULL);
  assert (ca != NULL);

  if (node->out_edges == NULL) { // base case
    assert (path_list->next == NULL);
    return;
  }
  
  if (best_path->next == NULL || path_list->next == NULL) {
    return;
  }
  
  assert(path_list != NULL);
  assert(best_path != NULL);

  double best_score = ca->best_current_time;
  
  assert(path_list->v < node->n_edges); 
  assert (node->out_edges[path_list->v].dst != NULL);
   
  // if this ant was on the best path
  if (best_path->v < node->n_edges && 
      node->out_edges[path_list->v].edge_id == node->out_edges[best_path->v].edge_id) {
    
    //printf("\t%0.4e\n", edge_eta(ant, &node->out_edges[path_list->v]));
    // deposit pheremone amount proportianal to the 
    // speed-up of this ant over the best timing
    double delta = (score > 10000 ? 0 : CA_Q * (best_score / score));
    node->out_edges[path_list->v].tau += delta;    
  }
  
  deposit_sched2(ant, node->out_edges[path_list->v].dst, path_list->next, best_path->next, 
		score, ca);
}

// not used?
struct path *deposit_sched(struct ca_node *node, struct path *path_list, struct path *best_path, 
		   int e_idx, double score, struct coding_ants *ca) {
  int i;
  printf("deposit %s\n", node->name); 
  assert (node != NULL);
  assert (best_path != NULL);
  assert (path_list != NULL);
  assert (ca != NULL);
  if (node->out_edges == NULL) {
    if (best_path->next == NULL ||
	path_list->next == NULL) {
      // the ant does not traverse the subgraph
      return path_list;
    }

    if (node->n_subgraph > 0) {
      struct path *plist = path_list;
      for (i = 0; i < node->n_subgraph; ++i) {
	plist = deposit_sched(node->subgraph[i].source, plist, best_path, e_idx + 1, score, ca);
      }
    }
    //assert(path_list->next == NULL);
    return path_list;
  }

  assert(path_list != NULL);
  assert(best_path != NULL);

  double best_score = ca->best_current_time;
  
  // if this ant was on the best path
  if (path_list->v == best_path->v) {
    // deposit pheremone amount proportianal to the 
    // speed-up of this ant over the best timing
    double delta = (score > 10000 ? 0.0 : CA_Q * (best_score / score));
    node->out_edges[path_list->v].tau += delta;    
  } 
  else {
    // this is a tree so as soon as they diverge
    // they will never be the same again
    return path_list;
  }
  
  // follow the next edge 
  return deposit_sched(node->out_edges[path_list->v].dst, path_list->next, best_path->next, 
		e_idx + 1, score, ca);
}

void deposit(struct ca_node *node, int *path, int *best_path, int e_idx, 
	     double score, struct coding_ants *ca) {
  int i;
  printf("deposit %s\n", node->name); 
  assert (node != NULL);
  assert (ca != NULL);
  if (node->out_edges == NULL) {
    return;
  }

  double best_score = ca->best_current_time;
  
  // if this ant was on the best path
  if (path[e_idx] == best_path[e_idx]) {
    // deposit pheremone amount proportianal to the 
    // speed-up of this ant over the best timing
    double delta = (score > 10000 ? 0.0 : CA_Q * (best_score / score));
    node->out_edges[path[e_idx]].tau += delta;    
  } 
  else {
    // this is a tree so as soon as they diverge
    // they will never be the same again
    return;
  }
  
  // follow the next edge 
  deposit(node->out_edges[path[e_idx]].dst, path, best_path, e_idx + 1, score, ca);
}

/*
 * Reset the edges tau to tau_start. This is 
 * needed because each ant decays the phermone value
 * along the edges it travels to encourage a more 
 * random search.
 */
void reset_tau(struct ca_node *node) {
  int i;
  //printf("r1\n");
  if (node == NULL || node->out_edges == NULL || node->visited) {
    return;
  }
  node->visited = 1;

  const int n_edges = node->n_edges;
  for (i = 0; i < n_edges; ++i) {
    node->out_edges[i].tau = node->out_edges[i].tau_start;
    reset_tau(node->out_edges[i].dst);
  }

  //node->visited = 0;  
}

void reset_visited(struct ca_node *node) {
  int i;
  //printf("r1\n");
  if (node == NULL || node->out_edges == NULL || !node->visited) {
    return;
  }
  node->visited = 0;

  const int n_edges = node->n_edges;
  for (i = 0; i < n_edges; ++i) {
    reset_visited(node->out_edges[i].dst);
  }

  //node->visited = 0;  
}

// normalize edges so that the pheremones weights sum to 1
void normalize_graph(struct ca_node *node) {
  int i;
  
  if (node == NULL || node->out_edges == NULL || node->visited) {
    return;
  }
  
  node->visited = 1;

  const int n_edges = node->n_edges;
  double sum_tau = 0.0;
  for (i = 0; i < n_edges; ++i) {
    sum_tau += node->out_edges[i].tau;
  }
  
  for (i = 0; i < n_edges; ++i) {
    //node->out_edges[i].tau /= sum_tau;
    
    // set the starting tau for the next iteration
    node->out_edges[i].tau_start = node->out_edges[i].tau;
  }

  for (i = 0; i < n_edges; ++i) {
    normalize_graph(node->out_edges[i].dst);
  }  

  //node->visited = 0;
}

void free_sched_path(struct path *path_list) {
  if (path_list->next == NULL) {
    return;
  }

  free_sched_path(path_list->next);
  free(path_list);
}

void update_graph(struct ca_ant *ants, struct coding_ants *ca, struct ca_ant best_ant) {
  int i, j;
  printf("updating graph...\n");
  reset_tau(ca->compiler_options_graph.source);
  reset_visited(ca->compiler_options_graph.source);
  decay(ca->compiler_options_graph.source);
  reset_visited(ca->compiler_options_graph.source);
  printf("u1\n");
  
  reset_tau(ca->mem_opt_graph.source);
  reset_visited(ca->mem_opt_graph.source);
  printf("u2\n");
  decay(ca->mem_opt_graph.source);  
  reset_visited(ca->mem_opt_graph.source);
  printf("u3\n");

  reset_tau(ca->sched_graph->source);
  reset_visited(ca->sched_graph->source);
 
  printf("u4\n");
  decay(ca->sched_graph->source);
  reset_visited(ca->sched_graph->source);
  printf("u5\n");

  deposit_best(ca->sched_graph->source, best_ant.s_path, ca->best_current_time);
  deposit_kernels(ca->sched_graph->source);
  for (i = 0; i < NUM_ANTS; ++i) {
    printf("deposit options\n");
    //deposit(ca->compiler_options_graph.source, ants[i].compiler_path, 
    //	    best_ant.compiler_path, 0, ants[i].score, ca);

    printf("deposit memory\n");
    //deposit(ca->mem_opt_graph.source, ants[i].mem_opt_path, 
    //	    best_ant.mem_opt_path, 0, ants[i].score, ca);

    printf("deposit schedule\n");
    //deposit_sched2(&ants[i], ca->sched_graph->source, ants[i].s_path, best_ant.s_path, 
    //		   ants[i].score, ca);

    printf("finished with deposit schedule\n");
  }
 //exit(0);
  printf("p1\n");
  // free the sched path list
  for (i = 0; i < NUM_ANTS; ++i) {
    free_sched_path(ants[i].s_path->next);
    ants[i].s_path->next = NULL;
    ants[i].s_path->v = -1;    
  }
  printf("p2\n");
  normalize_graph(ca->compiler_options_graph.source);
  reset_visited(ca->compiler_options_graph.source);
  printf("p3\n");
  normalize_graph(ca->mem_opt_graph.source);
  reset_visited(ca->mem_opt_graph.source);
  printf("p4\n");
  normalize_graph(ca->sched_graph->source);
  reset_visited(ca->sched_graph->source);
  printf("p5\n");
  printf("print graph\n");
  //print_graph_to_file("sched.graph", ca->sched_graph->source);  
  printf("finished update\n");
}

struct ca_ant *copy_ant(struct ca_ant ant) {
  int i, j;

  struct ca_ant *copy = (struct ca_ant *)calloc(1, sizeof(struct ca_ant));
  copy->s_path = (struct path *)calloc(1, sizeof(struct path));
  copy->s_path->next = NULL;

  struct path *trav = ant.s_path;
  struct path *cpy_trav = copy->s_path;
  assert (trav != NULL);

  while (trav != NULL) {  
    cpy_trav->v = trav->v;
    if (trav->next) {
      cpy_trav->next = (struct path *)calloc(1, sizeof(struct path));
      cpy_trav->next->next = NULL;
    }

    cpy_trav = cpy_trav->next;
    trav = trav->next;
  }
  assert (cpy_trav == NULL);

  copy->compiler_opts[0] = ant.compiler_opts[0];
  copy->compiler_path[0] = ant.compiler_path[0];

  copy->texs = (char **)calloc(ant.ntexs, sizeof(char*));
  copy->ntexs = ant.ntexs;

  for (i = 0; i < copy->ntexs; ++i) {
    copy->texs[i] = strdup(ant.texs[i]);
  }

  copy->score = ant.score;
  if (ant.sizes)
    copy->sizes = strdup(ant.sizes);
  else
    copy->sizes = NULL;

  if (ant.unroll_factors)
    copy->unroll_factors = strdup(ant.unroll_factors);
  else
    copy->unroll_factors = NULL;

  copy->opt_vec = (int *)calloc(num_regression_vec, sizeof(int));
  copy->opt_vec_ptr = ant.opt_vec_ptr;
  memcpy(copy->opt_vec, ant.opt_vec, num_regression_vec * sizeof(int));

  copy->kernel_stmt_id = ant.kernel_stmt_id;
  copy->kernel_cycles = ant.kernel_cycles;
  copy->kernel_stmts_len = (int *)calloc(ant.kernel_stmt_id, sizeof(int));
  copy->kernel_stmts = (char ***)calloc(ant.kernel_stmt_id, sizeof(char **));
  for (i = 0; i < ant.kernel_stmt_id; ++i) {
    copy->kernel_stmts_len[i] = ant.kernel_stmts_len[i];
    copy->kernel_stmts[i] = (char **)calloc(ant.kernel_stmts_len[i], sizeof(char *));

    for (j = 0; j < ant.kernel_stmts_len[i]; ++j) {
      copy->kernel_stmts[i][j] = strdup(ant.kernel_stmts[i][j]);
    }

  }
  
  return copy;
}

void save_current_best_schedule(struct coding_ants *ca) {
  FILE *fp;
  char from[100];
  char to[100];
  char command[200];

  char *prog_name = get_program_name(ca);

  strcpy(from, prog_name);
  strcpy(to, prog_name);
  strcat(from, "_ref.sched");
  strcat(to, "_best.sched");
  
  strcpy(command, "cp ");
  strcat(command, from);
  strcat(command, " ");
  strcat(command, to);

  fp = popen(command, "r");
  if (fp == NULL) {
    printf("Failed to run command\n" );
    exit(1);
  }
  pclose(fp);
}

int read_metrics_csv(struct ca_ant *ant) {
  int i;

  
  FILE *fp;
  const char *filename = "metrics.csv";
  if (!(fp = fopen(filename, "r"))) {
    fprintf(stderr, "ERROR opening profile csv\n");
    exit(1);
  }

  if (ant->nkernel <= 0) {
    return 0;
  }

  assert(ant->nkernel > 0);
  ant->kernel_metrics = (float **)calloc(ant->nkernel + 1, sizeof(float *));
  for (i = 0; i < ant->nkernel; ++i) {
    ant->kernel_metrics[i] = (float *)calloc(6, sizeof(float));
  }
  ant->kernel_cycles = (unsigned long *)calloc(ant->nkernel, sizeof(unsigned long));
  
  char line[1024];
  int line_num = 0;
  int metric_idx = 0;
  int gathered_data = 0;
  int metric_result = 0;
  if (ant->score < 1e6) {
    while(fgets(line, sizeof(line), fp)) {
      if (line_num++ <= 5)
	continue;
      
      if (!metric_result && strstr(line, "Metric result:")) {
	metric_result = 1;
      }

      char *kernel_str;
      if ((kernel_str = strstr(line, "kernel"))) {
	gathered_data = 1;
	int in_paren = 0;
	int kernel_id;
	int kernel_str_len = (strstr(kernel_str, "(") - kernel_str);

	char sid[5];
	memset(sid, 0, 5);
	memcpy(sid, kernel_str + 6, kernel_str_len - 6);
	kernel_id = atoi(sid);

	for (i = 0; i < sizeof(line); ++i) {
	  if (!line[i])
	    break;
        
	  if (line[i] == '(') {
	    ++in_paren;
	    continue;
	  }
	
	  if (line[i] == ')') {
	    --in_paren;
	    continue;
	  }

	  if(in_paren && line[i] == ',') {
	    line[i] = ' ';
	  }
	}

	// number of elements in split
	int line_len = 8;
      
	// contains the string 'kernel'
	char **split = str_split(line, ',');
      
	// factor to normalize the metric values
	float norm = 1.0f;
	if (strstr(split[line_len - 1], "MB/s")) {
	  norm = 1.0/1000.0f; // normalize to GB/s
	}
	else if(strstr(split[line_len - 1], "KB/s")) {
	  norm = 1.0/1000000.0f; // normalize to GB/s
	}
	else if(strstr(split[line_len - 1], "%")) {
	  norm = 1.0/100.0f; // normalize to 0 and 1
	}

	if (metric_result) {
	  float val = atof(split[line_len - 1]);
	  ant->kernel_metrics[kernel_id][metric_idx] = val * norm;
	  metric_idx = (metric_idx + 1) % 6;	
	} 
	else {
	  unsigned long val = strtoul(split[line_len - 1], NULL, 10);
	  assert (val > 0);
	  assert (kernel_id >= 0);
	  ant->kernel_cycles[kernel_id] = val;
	}	
      }
    }
  }
  
  int j;
  if (gathered_data) {
    for (i = 0; i < ant->nkernel; ++i) {
      //printf("kernel %d\n", i);
      ant->kernel_metrics[i][0] /= 8;
      ant->kernel_metrics[i][2] /= MAX_BANDWIDTH;
      ant->kernel_metrics[i][3] /= MAX_BANDWIDTH;
      for (j = 0; j < 6; ++j) {
	//	printf("%0.4e\n", ant->kernel_metrics[i][j]);
      }
      // printf("%lu\n", ant->kernel_cycles[i]); 
      //printf("\n");
    }
  }
  else {
    for (i = 0; i < ant->nkernel; ++i) {
      for (j = 0; j < 6; ++j) {
	// if it timed out, its really bad
	// set a small constant so that the ant
	// knows its bad
	ant->kernel_metrics[i][j] = 0.001f;
      }
      ant->kernel_cycles[i] = -1;
    }
  }

  fclose(fp);  
  return gathered_data;
}

/* Identifies the kernel that 'node' is optimizing. Returns the 
 * kernel id if all statments in 'node' belongs to a single kernel.
 * Returns -1 otherwise.
 */
int get_node_kernel(struct ca_ant *ant, struct ca_node *node) {
  int i, j;

  if (node->kernel_id >= 0) {
    return node->kernel_id;
  }

  if (node->stmt_str == NULL) {
    return -1;
  }

  int kernel = -1;
  for (i = 0; i < ant->kernel_stmt_id; ++i) {
    for (j = 0; j < node->stmt_len; ++j) {
      if (exists(node->stmt_str[j], ant->kernel_stmts[i], ant->kernel_stmts_len[i])) {
	if (kernel >= 0 && kernel != i) {
	  // node optimizes multiple kernels
	  return -1;
	}
	kernel = i;
      }
    }
  }

  return kernel;
}

void set_edge_metrics(struct ca_edge *edge, struct ca_ant *ant, int kernel, const int best) {
  int i, j, k;
  printf("dst name = %s\n", edge->dst->name);
  //assert (edge->metric_ptr < NUM_ANTS * NUM_ITER);
  assert(edge->has_been_traversed == 1);
  if (kernel >= ant->nkernel) {
    return;
  }

  float **kernel_metrics = ant->kernel_metrics;
  if (kernel < 0) {
    printf("weighted average metrics\n");
    unsigned long sum_cycles = 0;
    for (i = 0; i < ant->nkernel; ++i) {
      sum_cycles += ant->kernel_cycles[i];
    }
    
    for (i = 0; i < ant->nkernel; ++i) {
      double weight = (sum_cycles > 0 ? (double) ant->kernel_cycles[i] / sum_cycles : 0.01);
      for (j = 0; j < 6; ++j) {
	// the number of kernels with a non-zero jth metric
	int relevant_kernels = 0;
	for (k = 0; k < ant->nkernel; ++k) {
	  if (kernel_metrics[k][j] > 0.0f) {
	    ++relevant_kernels;
	  }
	}
	
	if (relevant_kernels > 0)
	  edge->metrics[edge->metric_ptr][j] += kernel_metrics[i][j] * weight;/// relevant_kernels;
      }
    }
    if (best) {      
      float sum = 0.0f;
      for (i = 0; i < 6; ++i)
	sum += edge->metrics[edge->metric_ptr][i];

      if (sum > 0)
	edge->best_metrics_id = edge->metric_ptr;
    }
    edge->metric_ptr++;
    return;
  }
  printf("d1\n");
  printf("kernel = %d\n", kernel);
  printf("metric ptr %d\n", edge->metric_ptr);
  printf("ant->nkernel = %d\n", ant->nkernel);
  printf("%f\n", edge->metrics[edge->metric_ptr][0]);
  printf("%f\n", kernel_metrics[kernel][0]);
  // single kernel
  for  (i = 0; i < 6; ++i) {
    printf("i = %d\n", i);
    edge->metrics[edge->metric_ptr][i] = kernel_metrics[kernel][i];
  }
  printf("d2\n");
  if (best) {    
    float sum = 0.0f;
    for (i = 0; i < 6; ++i) 
      sum += edge->metrics[edge->metric_ptr][i];
    
    if (sum > 0)
      edge->best_metrics_id = edge->metric_ptr;
  }
  printf("d3\n");
  ++edge->metric_ptr;  
}

void update_metrics(struct ca_ant *ant, struct ca_node *node, struct path *s_path, 
		    const int best) {
  int i, j;
  printf("update metrics %s\n", node->name);
  assert(node != NULL);
 
  node->best_score = (ant->score < node->best_score ? ant->score : node->best_score);
  if (node->out_edges == NULL || s_path == NULL) { // base case
    return;
  }
  assert(s_path != NULL);    

  struct ca_node *next = node->out_edges[s_path->v].dst;
  assert(next != NULL);
  
  int kernel = get_node_kernel(ant, next);
  // set the kernel score
  if (next->in_kernel && kernel >= 0) {
    assert (kernel >= 0);
    if (ant->kernel_cycles[kernel] < next->kernel_score && ant->kernel_cycles[kernel] > 0) {
      if (next->stmt_str == NULL) {
	next->stmt_str = copy_str_arr(ant->kernel_stmts[kernel],
				      ant->kernel_stmts_len[kernel]);
	next->stmt_len = ant->kernel_stmts_len[kernel];
      }
      next->kernel_score = ant->kernel_cycles[kernel];
      set_min_kern(ant, next);
    }
    //next->kernel_score = (ant->kernel_cycles[kernel] < next->kernel_score ? 
    //			  ant->kernel_cycles[kernel] :
    //			  next->kernel_score);

  }

  // add metrics to this edge
  set_edge_metrics(&node->out_edges[s_path->v], ant, kernel, best);
  
  update_metrics(ant, next, s_path->next, best);
}

// compute the l2 error between estimated and 
// observed metric values
float get_error(struct ca_ant *ant) {
  int i, j;

  float *v1 = ant->est_metrics;
  float *v2 = (float *)calloc(5, sizeof(float));
  
  for (i = 0; i < ant->kernel_stmt_id; ++i) {
    for (j = 0; j < 5; ++j) {
      v2[j] += (ant->kernel_metrics[i][j] / 5.0f);
    }
  }

  printf("est metric\n");
  float num = 0.f, den = 0.f;
  for (i = 0; i < 5; ++i) {
    //printf("%0.4e\n", v1[i]);
    num += ((v1[i] - v2[i]) * (v1[i] - v2[i]));
    den += (v2[i] * v2[i]);
  }
  
  num = sqrtf(num);
  den = sqrtf(den);

  return (num / (den == 0.0f ? 0.1 : den));
}

/* Compute the variance of the samples in 'samples' */
double variance(struct regression_sample *leaf, const int n_sample) {
  int i;
  double mean = 0.0;
  double var = 0.0;
  
  if (n_sample == 0)
    return 0.0;

  for (i = 0; i < n_sample; ++i) {
    mean += leaf[i].score;
  }

  mean /= n_sample;
  
  for (i = 0; i < n_sample; ++i) {
    var += ((leaf[i].score - mean) * (leaf[i].score - mean));
  }

  return var / n_sample;
}

double variance2(double *vec, const int n) {
  int i;
  double mean = 0.0;
  double var = 0.0;
  
  if (n == 0)
    return 0.0;

  int size = 0;
  for (i = 0; i < n; ++i) {
    if (vec[i] < 0)
      continue;

    ++size;
    mean += vec[i];
  }

  mean /= size;
  
  for (i = 0; i < n; ++i) {
    if (vec[i] < 0)
      continue;

    var += ((vec[i] - mean) * (vec[i] - mean));
  }
  
  return var / size;
}

double variance_plus(struct regression_sample *leaf, const int n_sample) {
  int i, j;

  double *scores = (double *)calloc(n_sample, sizeof(double));
  for (i = 0; i < n_sample; ++i)  
    scores[i] = leaf[i].score;

  double var = variance2(scores, n_sample);
  /*  double met_var = 0;
  for (j = 0; j < 6; ++j) {
    memset(scores, 0, n_sample * sizeof(double));
    for (i = 0; i < n_sample; ++i) {
      scores[i] = leaf[i].metric[j];
    }
    met_var += variance2(scores, n_sample);
  }
  
  met_var /= 6;
  */
  return var;//( var + met_var);
}

int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}

/* Return a sorted array of attribute values for the 'attr_id' attribute.
 * The array will not include duplicates. Sets len to the length of the 
 * returned array
 */
int *sorted_attributes(struct regression_sample *set, 
		       const int attr_id, const int n_sample, int *len) {
  assert (n_sample > 0);
  int i;
  int *attr_vec = (int *)calloc(n_sample, sizeof(int));
  for (i = 0; i < n_sample; ++i) {
    if (set[i].attr == NULL)
      continue;

    attr_vec[i] = set[i].attr[attr_id];
  }

  qsort(attr_vec, n_sample, sizeof(int), cmpfunc);
  
  int n_unique = 1;
  for (i = 1; i < n_sample; ++i) {
    if (attr_vec[i] != attr_vec[i-1]) {
      n_unique++;
    }
  }
  
  int *splits = (int *)calloc(n_unique, sizeof(int));
  splits[0] = attr_vec[0];
  int idx = 1;
  for (i = 1; i < n_sample; ++i) {
    if (attr_vec[i] != attr_vec[i-1]) {
      splits[idx++] = attr_vec[i];
    }    
  }
  
  *len = n_unique;
  free(attr_vec);
  return splits;
}

/* Compute the number of samples in 'set' that have 'attr_id' > split */
int rchild_size(struct regression_sample *set, const int n_sample, 
		const int attr_id, int split) {
  int i;
  int rsize = 0;
  for (i = 0; i < n_sample; ++i) {
    if (set[i].attr == NULL)
      continue;

    if (set[i].attr[attr_id] > split)
      ++rsize;
  }

  return rsize;
}

/* Compute the samples in 'set' that have 'attr_id' > split */
struct regression_sample* rchild(struct regression_sample *set, const int attr_id, 
				 const int n_sample, int split, int *len) {
  int i;
  struct regression_sample *rset;

  int rsize = 0;
  for (i = 0; i < n_sample; ++i) {
    if (set[i].attr == NULL)
      continue;

    printf("rchild = %d\n", set[i].attr[attr_id]);
    if (set[i].attr[attr_id] > split)
      ++rsize;
  }
  
  rset = (struct regression_sample *)calloc(rsize, sizeof(struct regression_sample));

  int idx = 0;
  for (i = 0; i < n_sample; ++i) {
    if (set[i].attr == NULL)
      continue;

    if (set[i].attr[attr_id] > split) {
      copy_regression_sample(&rset[idx++], &set[i]);            
    }    
  }

  *len = rsize;
  return rset;
}

/* Compute the samples in 'set' that have 'attr_id' <= split */
struct regression_sample* lchild(struct regression_sample *set, const int attr_id, 
				 const int n_sample, int split, int *len) {
  int i;
  struct regression_sample *lset;

  int lsize = 0;
  for (i = 0; i < n_sample; ++i) {
    if (set[i].attr == NULL)
      continue;

    printf("lchild = %d\n", set[i].attr[attr_id]);
    if (set[i].attr[attr_id] <= split)
      ++lsize;
  }
  
  lset = (struct regression_sample *)calloc(lsize, sizeof(struct regression_sample));

  int idx = 0;
  for (i = 0; i < n_sample; ++i) {
    if (set[i].attr == NULL)
      continue;

    if (set[i].attr[attr_id] <= split) {
      copy_regression_sample(&lset[idx++], &set[i]);            
    }    
  }

  *len = lsize;
  return lset;
}

/* Returns true if all 'attr_id' values are the same */
int single_attribute_val(struct regression_sample *set, const int attr_id, const int n_sample) {
  int i;
  if (set[0].attr == NULL)
    return 1;

  int ref = set[0].attr[attr_id];
  
  for (i = 1; i < n_sample; ++i) {
    if (set[i].attr == NULL)
      continue;

    if (ref != set[i].attr[attr_id]) {
      return 0;
    }
  }
  
  return 1;
}

/* Compute the sum of leaf variances for each split along 'attr_id' in 
 * 'set'. Return the smallest variance and set the 'split' variable to 
 * the corresponding split.
 */
double compute_min_split_variance(struct regression_sample *set, 
			      const int attr_id, const int n_sample, int *split) {

  int i, j, k;
  int len = 0;

  if (single_attribute_val(set, attr_id, n_sample)) {
    return 1e9;
  }

  int *split_cand = sorted_attributes(set, attr_id, n_sample, &len);
  assert (len > 0);
  
  if (len == 1) {
    *split = split_cand[0];
    return variance_plus(set, n_sample);
  }

  printf("%s\n", int_to_string(split_cand, len));
  printf("n_sample = %d\n", n_sample);
  double split_variance = -1.0;

  // upper bound is len - 1 because split on the last value
  // just results in the left child == set and right child empty
  for (i = 0; i < len - 1; ++i) {   // iterate over the possible splits
    // get the size of the left and right child
    int lsize, rsize;
    struct regression_sample *c1 = lchild(set, attr_id, n_sample, split_cand[i], &lsize);
    struct regression_sample *c2 = rchild(set, attr_id, n_sample, split_cand[i], &rsize);
    assert (lsize < n_sample);
    assert (rsize < n_sample);
    
    double var = variance_plus(c1, lsize) + variance_plus(c2, rsize);
    //    printf("min var = %0.2e, var = %0.2e, attr = %d, split = %d\n", split_variance, var, 
    //attr_id, split_cand[i]);
    for (j = 0; j < lsize; ++j) {
      //printf("\tlchild = %s, score = %0.2e\n", int_to_string(c1[j].attr, num_regression_vec), 
      //	     c1[j].score);
    }
    for (j = 0; j < rsize; ++j) {
      // printf("\trchild = %s, score = %0.2e\n", int_to_string(c2[j].attr, num_regression_vec),
      //     c2[j].score);
    }
    if(split_variance < 0 || split_variance > var) {
      split_variance = var;
      *split = split_cand[i];
    }
    
    // clean up
    free_regression_sample(c1, lsize);
    free_regression_sample(c2, rsize);
  } // end for i

  return split_variance;
}


/* Return true of all samples in leaf have the same attribute values */
int identical_samples(struct regression_sample *leaf, const int n_sample) {
  int i, j;
  
  struct regression_sample *ref = &leaf[0];
  if (leaf[0].attr == NULL)
    return 1;

  for (i = 1; i < n_sample; ++i) {
    for (j = 0; j < num_regression_vec; ++j) {
      if (leaf[i].attr == NULL)
	continue;

      if (ref->attr[j] != leaf[i].attr[j]) {
	return 0;
      }
    }
  }

  return 1;  
}

/* Split the samples defined by and generate a node 
 * defining the split. Then recurse on the two children.
 */
struct ca_node *regression_split(struct regression_sample *leaf, const int n_sample) {
  int i, j;
  double S = variance_plus(leaf, n_sample);
  if (n_sample <= 2) {
    return NULL; // base case
  }

  if (identical_samples(leaf, n_sample)) {
    return NULL;
  }

  double min_var = -1.0;
  int min_split = -1, min_attr = -1;
  // for each attribute
  for (i = 0; i < num_regression_vec; ++i) {
    int split = -1;
    double S_c = compute_min_split_variance(leaf, i, n_sample, &split);
    printf("S = %0.2e, Sc = %0.2e, split = %d\n", S, S_c, split);
    if (min_var < 0 || min_var > S_c) {
      min_var = S_c;
      min_split = split;
      min_attr = i;
    }
  }

  if (min_split < 0)
    return NULL;

  assert (min_split >= 0);

  if (S <= min_var) {
    //return NULL; // base case
  }

  printf("selected attr = %d, split = %d\n", min_attr, min_split);
  int lsize=0, rsize=0;
  struct regression_sample *c1 = lchild(leaf, min_attr, n_sample, min_split, &lsize);
  struct regression_sample *c2 = rchild(leaf, min_attr, n_sample, min_split, &rsize);
  
  printf("lsize = %d, n_sample %d\n", lsize, n_sample);
  assert (lsize < n_sample);
  assert (rsize < n_sample);

  assert (lsize > 0);
  assert (rsize > 0);

  char name[50];
  snprintf(name, 50, "attr %d <= %d", min_attr, min_split);
  struct node_split *ns = (struct node_split *)calloc(1, sizeof(struct node_split));
  ns->attr = min_attr;
  ns->split = min_split;
  struct ca_node *node = create_node(REGRESSION, name, ns);
  ca_allocate_out_edges(node, 2);
  
  node->out_edges[0].dst = regression_split(c1, lsize); 
  node->out_edges[1].dst = regression_split(c2, rsize); 

  //printf("var = %0.2e, split %d on attr %d\n", min_var, min_split, min_attr); 
  return node;
}

/* Return either 0 or 1 to distinguish whether opt_vec
 * matches the left or right child of the split
 */
int which_path(int *opt_vec, struct node_split *ns) {
  const int attr = ns->attr;
  const int split = ns->split;
 
  if (opt_vec[attr] <= split) {
    return 0; // left path
  }
  return 1; // rigth path
}

void deposit_regression_tree(struct regression_sample sant, 
			     struct ca_ant *best_ant,
			     struct ca_node *node) {
  int i;  
  if (node == NULL || node->out_edges == NULL) {
    return;
  }

  if (sant.attr == NULL)
    return;

  //printf("deposit reg %s\n", node->name);
  assert (node->type == REGRESSION);
  
  struct node_split *ns = (struct node_split *)node->val;

  int ant_path = which_path(sant.attr, ns);
  int best_path = which_path(best_ant->opt_vec, ns);

  if (ant_path != best_path) {
    // its a tree so just quit
    return;
  }
  
  double score = sant.score;
  double best_score = best_ant->score;

  double delta = (score > 10000 ? 0 : CA_Q * (best_score / score));
  node->out_edges[ant_path].tau += delta; 
  node->out_edges[ant_path].tau = (node->out_edges[ant_path].tau > MAX_TAU ? MAX_TAU :
				   node->out_edges[ant_path].tau);   
  
  deposit_regression_tree(sant, best_ant, node->out_edges[ant_path].dst);
}

void deposit_reg_tree(struct ca_node *node, 
		      struct regression_sample sant, double best_score) {

  int i;  
  if (node == NULL || node->out_edges == NULL) {
    return;
  }

  if (sant.attr == NULL)
    return;

  //printf("deposit reg %s\n", node->name);
  assert (node->type == REGRESSION);
  
  struct node_split *ns = (struct node_split *)node->val;

  int ant_path = which_path(sant.attr, ns);
    
  double score = sant.score;
  
  double delta = (score > 10000 ? 0 : CA_Q * (best_score / score));
  node->out_edges[ant_path].tau += delta; 
  //node->out_edges[ant_path].tau = (node->out_edges[ant_path].tau > MAX_TAU ? MAX_TAU :
  //				   node->out_edges[ant_path].tau);   
  
  struct ca_edge *edge = &(node->out_edges[ant_path]);
  for (i = 0; i < 6; ++i) {
    edge->metrics[edge->metric_ptr][i] = sant.metric[i];
  }
  ++edge->metric_ptr;

  deposit_reg_tree(node->out_edges[ant_path].dst, sant, best_score);
}

// Make the ant select the 'best' ants path. This will always be the one 
// with the most pheremone
int *regression_best_filter(struct ca_node *node) {
  int i;

  const int n_edges = node->n_edges;  
  int *blacklist = (int *)calloc(n_edges, sizeof(int));

  int best_path = -1;
  double max_tau = 0;

  for (i = 0; i < n_edges; ++i) {
    double tau = node->out_edges[i].tau;
    if (tau > max_tau) {
      best_path = i;
      max_tau = tau;
    }
  }

  for (i = 0; i < n_edges; ++i) {
    if (i == best_path)
      continue;

    blacklist[i] = 1;
  }

  return blacklist;
}

void traverse_regression_tree(struct ca_ant *ant, struct ca_node *node, int force_best_path) {
  if (node == NULL || node->out_edges == NULL) {
    return;
  }
  assert (node->type == REGRESSION);

  struct node_split *ns = (struct node_split *)node->val;
  
  int *filter = (force_best_path ? regression_best_filter(node) : NULL);

  int e_idx = select_edge(ant, node, filter);

  if (!force_best_path) {
    // decay edge to encourage more randomness
    //decay_edge(&(node->out_edges[e_idx]));
  }

  if (filter) {
    free(filter);
    filter = NULL;
  }

  assert (e_idx < 2);
  if (e_idx == 0) {
    // must be <= than split
    ant->filter.attr_ub[ns->attr] = ns->split;
  }
  else {
    // must be >= split + 1
    ant->filter.attr_lb[ns->attr] = ns->split + 1;
  }

  //ant->filter.attr_vec[ns->attr] = ns->split;
  //ant->filter.lte_mask[ns->attr] = e_idx; // 0 -> lte, 1 -> gt 
  
  traverse_regression_tree(ant, node->out_edges[e_idx].dst, force_best_path);
}

void reset_regression_tau(struct ca_node *node) {
  if (node == NULL || node->out_edges == NULL) {
    return;
  }
  
  node->out_edges[0].tau = 0.5;
  node->out_edges[1].tau = 0.5;
  
  reset_regression_tau(node->out_edges[0].dst);
  reset_regression_tau(node->out_edges[1].dst);
}

void decay_regression_tau(struct ca_node *node) {
  if (node == NULL || node->out_edges == NULL) {
    return;
  }
  
  decay_edge(&(node->out_edges[0]));
  decay_edge(&(node->out_edges[1]));

  decay_regression_tau(node->out_edges[0].dst);
  decay_regression_tau(node->out_edges[1].dst);
}


void deposit_reg_tree_all(struct ca_node *root, const int n_sample, double best_score) {
  assert (n_sample > 0);
  assert (best_score > 0);
  int i;
  for (i = 0; i < n_sample; ++i) {
    if (i > 0 && (i % NUM_ANTS) == 0) {
      decay_regression_tau(root);
    }
    deposit_reg_tree(root, samples[i], best_score);
  }  
  print_graph_to_file("regression.graph", root);  
}

struct ca_node *build_regression_tree(struct ca_ant *best, const int n_sample) {
  printf("build regression tree\n");
  struct ca_node *root = regression_split(samples, n_sample);
  printf("end build regression tree\n");
  return root;
}

void update_regression_tree_edges(struct ca_node *root, struct ca_ant *best, const int n_sample) {
  printf("update regression tree\n");
  int i, j;
  int iteration = n_sample / NUM_ANTS;
  for (i = 0; i < iteration; ++i) {
    decay(root);
    reset_visited(root);
    for (j = 0; j < NUM_ANTS; ++j) {
      if (samples[i * NUM_ANTS + j].attr == NULL)
	continue;

      deposit_regression_tree(samples[i * NUM_ANTS + j], best, root);
    }
    normalize_graph(root);
    reset_visited(root);
  }
  print_graph_to_file("regression.graph", root);  
  printf("end update regression tree\n");
}

/* Check that the selected optimizations respect the 
 * constraints set by the regression tree traversal
 */
int satisfies_filter(struct ca_ant *ant) {
  int i;
  if (!ant->filter_set) {
    // trivially satisfied
    return 1; 
  }
  
  for (i = 0; i < num_regression_vec; ++i) {
    int lb = ant->filter.attr_lb[i];
    int ub = ant->filter.attr_ub[i];

    if (i >= tile_start && i < unroll_start) {
      // this may not always be possible to satisfy
      continue;
    }

    if (i >= unroll_start) {
      // no unrolling took place because
      // there is an additional layer of 
      // parallelism, which does not get unrolled
      if (ant->opt_vec[i] == 0) {
	continue;
      }

      // it may be impossible to satisfy
      // the lower and upper bounds simultaneously.
      if (lb >= 0 && ub >= 0) {
	continue;
      }

      // fusion alone can violate the upper bound
      if (ub >= 0) {
	continue;
      }
      continue;
    }

    if (i == reduction_start) {
      if (ub == 0) {
	if (ant->opt_vec[i] > 0) 
	  return 0;
      }

      // don't care about the lower bound
      continue;
    }

    if (ub >= 0) {
      // must be <= ub
      if (ant->opt_vec[i] > ub)
	return 0;

    }
    
    if (lb >= 0) {
      // must be >= lb
      if (ant->opt_vec[i] < lb)
	return 0;
    }
  }

  return 1;  
}

struct dfs_sample {
  int *pos;
  int n;

  int *tree_pos;
  int tree_len;

  double runtime;
  struct dfs_sample *next;
};

struct best_dfs_sample {
  struct dfs_sample *sample;
  struct best_dfs_sample *next;
};

double percent_deviation(double orig, double num) {
  return ((num - orig) / orig) * 100.0;
}

double euclidean_distance(struct dfs_sample *d1, struct dfs_sample *d2) {
  assert (d1->n == d2->n);
  int i;
  const int n = d1->n;
  double dist = 0;
  for (i = 0; i < n; ++i) {
    dist += (d1->pos[i] -  d2->pos[i]) * (d1->pos[i] -  d2->pos[i]);
  }
  return sqrt(dist);
}

int lowest_common_ancestor(struct dfs_sample *d1, struct dfs_sample *d2) {
  int i;
  const int n = (d1->tree_len < d2->tree_len ? d1->tree_len : d2->tree_len);
  for (i = 0; i < n; ++i) {
    if (d1->tree_pos[i] != d2->tree_pos[i])
      return i;
  }

  return n;
}

int tree_distance(struct dfs_sample *d1, struct dfs_sample *d2) {
  int dist_1 = d1->tree_len;
  int dist_2 = d2->tree_len;
  
  int lca = lowest_common_ancestor(d1, d2);
  assert (lca >= 0);

  return (dist_1 + dist_2 - 2*lca);
}

int hamming_distance(struct dfs_sample *d1, struct dfs_sample *d2) {
  assert (d1->n == d2->n);
  
  int i;
  int n = d1->n;
  int dist = 0;
  for (i = 0; i < n; ++i) {
    dist += abs((d1->pos[i] - d2->pos[i]));
  }

  return dist;
}

int weighted_hamming_distance(struct dfs_sample *d1, struct dfs_sample *d2) {
  assert (d1->n == d2->n);
  
  int i;
  int n = d1->n;
  int dist = 0;
  for (i = 0; i < n; ++i) {
    dist += (n - i) * abs((d1->pos[i] - d2->pos[i]));
  }

  return dist;
}

/* If the sample defined by pos exists then return the runtime, else return -1 */
double get_dfs_runtime(struct dfs_sample *samples, int *pos, const int n) {
  struct dfs_sample *trav = samples;  
  while(trav->next != NULL) {
    if (trav->n == n) {
      if (vec1_eq_vec2(trav->pos, pos, n)) {
	return trav->runtime;
      }
    }
    
    trav = trav->next;
  }
  
  return -1.0;
}

/* get the convex diameter along each dimension with the center at best */
int* get_convex_diameter(struct dfs_sample *samples, struct dfs_sample *best) {
  int i;
  int n = best->n;
  int *center = int_copy(best->pos, n);
  int *convex = (int *)calloc(n, sizeof(int));

  for (i = 0; i < n; ++i) {
    double runtime = best->runtime;
    
    int *pt = int_copy(center, n);
    // search in the positve direction
    double score = runtime;

    int dist = 0;
    while (score >= 0.0) {
      pt[i]++;
      double new_score = get_dfs_runtime(samples, pt, n);
      if (new_score < score) {
	// not convex
	break;
      }
      
      if (new_score >= 0)
	++dist;

      score = new_score;
    } 
    free(pt);
    
    pt = int_copy(center, n);
    score = runtime;
    
    int neg_dist = 0;
    while (score >= 0.0) {
      pt[i]--;
      double new_score = get_dfs_runtime(samples, pt, n);
      if (new_score < score) {
	// not convex
	break;
      }
      
      if (new_score >= 0)
	++dist;

      score = new_score;
    } 
    free(pt);

    convex[i] = dist;
  } 
  free(center);

  return convex;
}

double mean (double *x, const int n) {
  int i;
  double mu = 0.0;
  for (i = 0; i < n; ++i) {
    mu += x[i];
  }
  return mu / n;
}

double covariance(double *X, double *Y, const int n) {
  int i;
  double ex = mean(X, n);
  double ey = mean(Y, n);

  double d = 0.0;
  for (i = 0; i < n; ++i) {
    d += ((X[i] - ex) * (Y[i] - ey));
  }
  
  return d / n;
}

double FDC(double *F, double *D, const int n) {
  double cov = covariance(F, D, n);
  double vF = variance2(F, n);
  double vD = variance2(D, n);

  return cov / (sqrt(vF) * sqrt(vD));
}


int *tree_vec(struct ca_ant *ant, int *pos, int *len, struct ca_node *node) {
  if (node == NULL || node->out_edges == NULL) {
    return pos;
  }
  assert (node->type == REGRESSION);
  struct node_split *ns = (struct node_split *)node->val;
  
  int *opt = ant->opt_vec;
  int attr = ns->attr;
  int split = ns->split;
  
  int v = (opt[attr] <= split ? 0 : 1);
  pos[(*len)++] = v;

  return tree_vec(ant, pos, len, node->out_edges[v].dst);
}

int *get_tree_pos(struct ca_ant *ant, struct ca_node *tree, int *len) {
  int *pos = (int *)calloc(100, sizeof(int));
  *len = 0;
  return tree_vec(ant, pos, len, tree);
}

// search the full space via depth first search
int optimize_random(isl_ctx *ctx,  struct ppcg_options *options,
		 const char *input, const char *output, 
		 __isl_give isl_printer *(*print_cuda)(__isl_take isl_printer *p,
		 struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
		 struct gpu_types *types, void *user),struct coding_ants *ca,
		 struct ca_node *regression_tree) {

  int i, j;
  random_search = 1;

  struct ca_wrapper wrap; 
  wrap.o2_failed = 0;
  wrap.traverse_partition = traverse_partition_graph;
  wrap.traverse_schedule = traverse_schedule_graph;
  wrap.traverse_reduction = traverse_reduction_graph;
  wrap.traverse_texture = traverse_texture_graph;
  wrap.traverse_tile = traverse_tile_sizes;
  wrap.traverse_inner_tile = traverse_inner_tile;
  wrap.traverse_unroll = traverse_unroll_factors;
  wrap.traverse_atomic_store = traverse_atomic_store;
  wrap.traverse_cache_config = traverse_cache_config;
  wrap.traverse_shared_mem = traverse_shared_memory;
  wrap.optimization = 1; // O2 set schedule and partitioning 
  wrap.reducible = NULL;

  wrap.nread_only = 0;
  wrap.read_only = get_read_only_arrays(ca, &wrap.nread_only);
  ca->wrap = &wrap;

  // use the dfs stack to record the path taken
  dfs_stack = (int *)calloc(DFS_STACK_SIZE, sizeof(int));
  
  // (1) initialize the ants
  struct ca_ant *ant = init_ants(ca, wrap.nread_only);
  ca->sched_graph = (struct ca_graph *)calloc(1, sizeof(struct ca_graph));
  ca->sched_graph->source = create_node(SOURCE, "source", NULL);
  int r;
  double best_time = 10000;
  int neval = 50;

  struct dfs_sample *samples = (struct dfs_sample *)calloc(1, sizeof(struct dfs_sample));
  samples->next = NULL;
  struct dfs_sample *samp = samples; 
  
  int max_dim = 0;

  for (i = 0; i < neval; ++i) {
    is_new_sample = 0;

    memset(dfs_stack, 0, DFS_STACK_SIZE * sizeof(int));
    dfs_stack_ptr = 0;

    ca->wrap->node = ca->sched_graph->source;
    ca->wrap->ant = ant;
    ca->wrap->s_path = ant->s_path;
    
    // select a path through the graph
    ca_traverse(ant, ca);
    pre_ca_apply(ant, ca);

    // generate the code
    r = generate_code(ctx, options, input, output, print_cuda, ca);
    
    printf("edges = %s\n", int_to_string(dfs_stack, dfs_stack_ptr));   

    printf("opt vec = %s\n", int_to_string(ant->opt_vec, num_regression_vec));    
    if (get_dfs_runtime(samples, dfs_stack, dfs_stack_ptr) >= 0) {//!is_new_sample) {
      --i;
      post_ca_apply(ant, ca);
      continue;
    }

    double runtime = evaluate(ca, 0);
    
    if (runtime >= 8) {
      --i;
      post_ca_apply(ant, ca);
      continue;
    }

    if (runtime <= 0.0001) {
      --i;
      post_ca_apply(ant, ca);
      continue;
    }
    if (runtime < best_time) {
      best_time = runtime;
    }

    max_dim = (max_dim < dfs_stack_ptr ? dfs_stack_ptr : max_dim);
    samp->pos = int_copy(dfs_stack, dfs_stack_ptr);
    samp->tree_pos = get_tree_pos(ant, regression_tree, &(samp->tree_len));
    samp->n = dfs_stack_ptr;
    
    printf("tree vec = %s\n", int_to_string(samp->tree_pos, samp->tree_len));
    samp->runtime = runtime;
    samp->next = (struct dfs_sample *)calloc(1, sizeof(struct dfs_sample));
    samp->next->next = NULL;
    samp = samp->next;

    post_ca_apply(ant, ca);
    printf("eval %d, runtime = %0.4e, best so far = %0.4e\n", i, runtime, best_time);   
  }

  // test point
  /*samp->pos = (int *)calloc(5, sizeof(int));
  samp->n = 5;
  samp->runtime = 0.09;
  samp->next = (struct dfs_sample *)calloc(1, sizeof(struct dfs_sample));
  samp->next->next = NULL;
  samp = samp->next;
  */

  struct best_dfs_sample **best_samp = 
    (struct best_dfs_sample **)calloc(max_dim + 1, sizeof(struct best_dfs_sample *));

  for (i = 0; i < max_dim + 1; ++i) {
    //    best_samp[i]->sample = NULL;
    //best_samp[i]->next = NULL;
  }
  
  struct dfs_sample *trav = samples;
  while(trav->next != NULL) {
    const int n = trav->n;
    if (!best_samp[n]) {
      best_samp[n] = (struct best_dfs_sample *)calloc(1, sizeof(struct best_dfs_sample));
    }

    if (best_samp[n]->sample == NULL || trav->runtime < best_samp[n]->sample->runtime) {
      best_samp[n]->sample = trav;
    }
    trav = trav->next;
  }

  struct best_dfs_sample **best_travs = 
    (struct best_dfs_sample **)calloc(max_dim + 1, sizeof(struct best_dfs_sample *));
  
  for (i = 0; i < max_dim + 1; ++i) {
    best_travs[i] = best_samp[i];
  }
  
  trav = samples;
  while (trav->next != NULL) {
    int n = trav->n;
    if (best_samp[n]->sample != trav) {
      double dev = best_samp[n]->sample->runtime * 0.075;
      if (trav->runtime <= best_samp[n]->sample->runtime + dev) {
	//best_samp[n].next = trav;
	best_travs[n]->next = (struct best_dfs_sample *)calloc(1, sizeof(struct best_dfs_sample));
	best_travs[n]->next->next = NULL;
	best_travs[n]->next->sample = trav;
	best_travs[n] = best_travs[n]->next;
	//printf("%f, %f\n", best_samp[n].sample->runtime, trav->runtime);
	//printf("HERE\n");
	//exit(0);
      }
    }
    trav = trav->next;
  }
  
  for (i = 0; i < max_dim + 1; ++i) {
    struct best_dfs_sample *btrav = best_samp[i];
    while (btrav != NULL) {
      printf("pos %s, runtime %f\n", int_to_string(btrav->sample->pos, btrav->sample->n), 
	     btrav->sample->runtime);
      btrav = btrav->next;
    }    
  }

  char *corr_name = get_program_name(ca);
  strcat(corr_name, "_fdc.txt");

  FILE *fp_corr = fopen(corr_name, "w");
  if (fp_corr == NULL) {
    printf("Failed to open file\n");
    exit(1);
  }

  fprintf(fp_corr, "# runtime, hamming distance, ham perc dev., weighted ham distance, weighted perc dev., euclidean, perc dev.\n");

  // fitness vector
  double *F = (double *)calloc(neval, sizeof(double));
  // distance vector
  double *D = (double *)calloc(neval, sizeof(double));
  // weighted distance vector
  double *wD = (double *)calloc(neval, sizeof(double));
  // euclidean distance vector
  double *eucD = (double *)calloc(neval, sizeof(double));
  // regression tree distance vector
  double *tD = (double *)calloc(neval, sizeof(double));

  int ct = 0;
  trav = samples;
  while(trav->next != NULL) {
    const int n = trav->n;
    int min_x = -1;
    int min_wx = -1;
    double min_euc = -1;
    int min_tx = -1;

    int x = -1;
    int wx = -1;
    double euc_x = -1;
    int tx = -1;
    struct dfs_sample *closest = NULL;
    struct dfs_sample *w_closest = NULL;
    struct dfs_sample *euc_closest = NULL;
    struct dfs_sample *t_closest = NULL;

    // get the closest best sample
    struct best_dfs_sample *btrav = best_samp[n];
    while (btrav != NULL) {
      x = hamming_distance(btrav->sample, trav);
      if (min_x < 0 || x < min_x) {
	  min_x = x;
	  closest = btrav->sample;
      }
      wx = weighted_hamming_distance(btrav->sample, trav);
      if (min_wx < 0 || wx < min_wx) {
	min_wx = wx;
	w_closest = btrav->sample;
      }

      euc_x = euclidean_distance(btrav->sample, trav);
      if (min_euc < 0 || euc_x < min_euc) {
	min_euc = euc_x;
	euc_closest = btrav->sample;
      }
      
      tx = tree_distance(btrav->sample, trav);
      if (min_tx < 0 || tx < min_tx) {
	min_tx = tx;
        t_closest = btrav->sample;
      }
      btrav = btrav->next;
    } // end while btrav
    
    x = min_x;
    wx = min_wx;
    euc_x = min_euc;
    tx = min_tx;

    assert (x >= 0);
    assert (wx >= 0);
    assert (euc_x >= 0);
    assert (tx >= 0);

    assert (closest);
    assert (w_closest);
    assert (euc_closest);
    assert (t_closest);

    double y = percent_deviation(closest->runtime, trav->runtime);
    double wy = percent_deviation(w_closest->runtime, trav->runtime);
    double euc_y = percent_deviation(euc_closest->runtime, trav->runtime);
    double ty = percent_deviation(t_closest->runtime, trav->runtime);
    F[ct] = trav->runtime;
    wD[ct] = wx;
    tD[ct] = tx;
    eucD[ct] = euc_x;
    D[ct++] = x;
    //printf("x = %d, y = %0.4e\n", x, y);

    //fprintf(fp_corr, "%f, %d, %f, %d, %f, %f, %f, %d, %f\n", trav->runtime, x, y, wx, wy, 
    //	    euc_x, euc_y, tx, ty);
    fprintf(fp_corr, "%f\n", trav->runtime);

    trav = trav->next;
  } // end while trav

  double fdc = FDC(F, D, ct);
  double wfdc = FDC(F, wD, ct);
  double euc_fdc = FDC(F, eucD, ct);
  double tfdc = FDC(F, tD, ct);
  fprintf(fp_corr, "ppcg time = %f\n", ca->ppcg_only_time);
  printf("FDC = %f, wFDC = %f, eucFDC = %f, tFDC = %f\n", fdc, wfdc, euc_fdc, tfdc);
  fprintf(fp_corr, "# FDC = %f, wFDC = %f, eucFDC = %f, tFDC = %f\n", fdc, wfdc, euc_fdc, tfdc);

  free(F);
  free(D);
  free(wD);
  free(tD);

  for (i = 0; i < max_dim + 1; ++i) {
    struct best_dfs_sample *btrav = best_samp[i];
    if (btrav != NULL) {
      while(btrav != NULL) {
	fprintf(fp_corr, "# best = %s, %0.4e\n", 
		int_to_string(btrav->sample->pos, btrav->sample->n), btrav->sample->runtime);

	printf("# best = %s, %0.4e\n", 
		int_to_string(btrav->sample->pos, btrav->sample->n), btrav->sample->runtime);

	btrav = btrav->next;
      }
    }
  }
  fclose(fp_corr);

  
  /*
  char *corr_name = get_program_name(ca);
  strcat(corr_name, "_fdc.txt");

  FILE *fp_corr = fopen(corr_name, "w");
  if (fp_corr == NULL) {
    printf("Failed to open file\n");
    exit(1);
  }
  
  // fitness vector
  double *F = (double *)calloc(neval, sizeof(double));
  // distance vector
  double *D = (double *)calloc(neval, sizeof(double));
  // weighted distance vector
  double *wD = (double *)calloc(neval, sizeof(double));
  int ct = 0;
  trav = samples;
  while(trav->next != NULL) {
    int n = trav->n;
    int min_id = 0;    
    int min_x = -1;
    int min_wid = 0;    
    int min_wx = -1;
    int x = -1;
    int wx = -1;
    for (i = 0; i < 3; ++i) {
      if (best_samp[n][i]) {
	x = hamming_distance(best_samp[n][i], trav);
	if (min_x < 0 || x < min_x) {
	  min_x = x;
	  min_id = i;
	}
	wx = weighted_hamming_distance(best_samp[n][i], trav);
	if (min_wx < 0 || wx < min_wx) {
	  min_wx = wx;
	  min_wid = i;
	}
      }
    }

    x = min_x;
    wx = min_wx;
    assert (x >= 0);
    assert (min_id >= 0);
    assert (wx >= 0);
    assert (min_wid >= 0);
    
    
    //int x = hamming_distance(best_samp[n][0], trav);
    //int wx = weighted_hamming_distance(best_samp[n][min_id], trav);
    double y = percent_deviation(best_samp[n][min_id]->runtime, trav->runtime);
    double wy = percent_deviation(best_samp[n][min_wid]->runtime, trav->runtime);
    F[ct] = trav->runtime;
    wD[ct] = wx;
    D[ct++] = x;
    //printf("x = %d, y = %0.4e\n", x, y);

    fprintf(fp_corr, "%d, %f, %d, %f\n", x, y, wx, wy);
    trav = trav->next;
  }
  double fdc = FDC(F, D, ct);
  double wfdc = FDC(F, wD, ct);
  printf("FDC = %f, wFDC = %f\n", fdc, wfdc);
  fprintf(fp_corr, "# FDC = %f, wFDC = %f\n", fdc, wfdc);

  free(F);
  free(D);
  free(wD);

  for (i = 0; i < max_dim + 1; ++i) {
    if (best_samp[i][0]) {
      fprintf(fp_corr, "# best = %s, %0.4e\n", int_to_string(best_samp[i][0]->pos, best_samp[i][0]->n), 
	      best_samp[i][0]->runtime);
    }
  }

  fclose(fp_corr);

  for (i = 0; i < max_dim + 1; ++i) {
    if (best_samp[i][0]) {
      printf("best = %s, %0.4e\n", int_to_string(best_samp[i][0]->pos, best_samp[i][0]->n), best_samp[i][0]->runtime);
    }

    if (best_samp[i][1]) {
      printf("best1 = %s, %0.4e\n", int_to_string(best_samp[i][1]->pos, best_samp[i][1]->n), best_samp[i][1]->runtime);
    }


    if (best_samp[i][2]) {
      printf("best2 = %s, %0.4e\n", int_to_string(best_samp[i][2]->pos, best_samp[i][2]->n), best_samp[i][2]->runtime);
    }
    }*/
  return 0;
}

// search the full space via depth first search
int optimize_dfs(isl_ctx *ctx,  struct ppcg_options *options,
		 const char *input, const char *output, 
		 __isl_give isl_printer *(*print_cuda)(__isl_take isl_printer *p,
		 struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
	         struct gpu_types *types, void *user),struct coding_ants *ca) {

  int i, j;
  depth_first_search = 1;

  struct ca_wrapper wrap; 
  wrap.o2_failed = 0;
  wrap.traverse_partition = traverse_partition_graph;
  wrap.traverse_schedule = traverse_schedule_graph;
  wrap.traverse_reduction = traverse_reduction_graph;
  wrap.traverse_texture = traverse_texture_graph;
  wrap.traverse_tile = traverse_tile_sizes;
  wrap.traverse_inner_tile = traverse_inner_tile;
  wrap.traverse_unroll = traverse_unroll_factors;
  wrap.traverse_atomic_store = traverse_atomic_store;
  wrap.traverse_cache_config = traverse_cache_config;
  wrap.traverse_shared_mem = traverse_shared_memory;
  wrap.optimization = 1; // O2 set schedule and partitioning 
  wrap.reducible = NULL;

  wrap.nread_only = 0;
  wrap.read_only = get_read_only_arrays(ca, &wrap.nread_only);
  ca->wrap = &wrap;

  dfs_stack = (int *)calloc(DFS_STACK_SIZE, sizeof(int));
  
  // (1) initialize the ants
  struct ca_ant *ant = init_ants(ca, wrap.nread_only);
  ca->sched_graph = (struct ca_graph *)calloc(1, sizeof(struct ca_graph));
  ca->sched_graph->source = create_node(SOURCE, "source", NULL);
  int r;
  double best_time = 10000;
  int neval = 0;

  struct dfs_sample *samples = (struct dfs_sample *)calloc(1, sizeof(struct dfs_sample));
  samples->next = NULL;
  struct dfs_sample *samp = samples; 
  
  int max_dim = 0;
  push(0);
  do {

    current_depth = 0;
    
    ca->wrap->node = ca->sched_graph->source;
    ca->wrap->ant = ant;
    ca->wrap->s_path = ant->s_path;
    
    dfs_full_explored = dfs_full_explored_n;
    dfs_full_explore_depth = dfs_full_explore_depth_n;

    if (dfs_full_explored && dfs_full_explore_depth == 0)
      break;

    // select a path through the graph
    ca_traverse(ant, ca);
    pre_ca_apply(ant, ca);

    // generate the code
    r = generate_code(ctx, options, input, output, print_cuda, ca);

    printf("edges = %s\n", int_to_string(dfs_stack, dfs_stack_ptr));   

    double runtime = evaluate(ca, 0);
    if (runtime < best_time) {
      best_time = runtime;
    }

    max_dim = (max_dim < dfs_stack_ptr ? dfs_stack_ptr : max_dim);

    samp->pos = int_copy(dfs_stack, dfs_stack_ptr);
    samp->n = dfs_stack_ptr;
    
    samp->runtime = runtime;
    samp->next = (struct dfs_sample *)calloc(1, sizeof(struct dfs_sample));
    samp->next->next = NULL;
    samp = samp->next;

    post_ca_apply(ant, ca);
    printf("eval %d, runtime = %0.4e, best so far = %0.4e\n",neval, runtime, best_time); 
    ++neval;
  } while(dfs_stack_ptr > 0 && neval < 2048);

  // test point
  /*samp->pos = (int *)calloc(5, sizeof(int));
  samp->n = 5;
  samp->runtime = 0.09;
  samp->next = (struct dfs_sample *)calloc(1, sizeof(struct dfs_sample));
  samp->next->next = NULL;
  samp = samp->next;
  */

  struct dfs_sample **best_samp = (struct dfs_sample **)calloc(max_dim + 1, 
							       sizeof(struct dfs_sample *));
  
  //struct dfs_sample *best_samp = NULL;
  for (i = 0; i < max_dim + 1; ++i) {
    best_samp[i] = NULL;
  }
  
  struct dfs_sample *trav = samples;
  while(trav->next != NULL) {
    if (best_samp[trav->n] == NULL || trav->runtime < best_samp[trav->n]->runtime) {
      best_samp[trav->n] = trav;
    }
    trav = trav->next;
  }
  
  char *corr_name = get_program_name(ca);
  strcat(corr_name, "_fdc.txt");

  FILE *fp_corr = fopen(corr_name, "w");
  if (fp_corr == NULL) {
    printf("Failed to open file\n");
    exit(1);
  }
  
  // fitness vector
  double *F = (double *)calloc(neval, sizeof(double));
  // distance vector
  double *D = (double *)calloc(neval, sizeof(double));
  // weighted distance vector
  double *wD = (double *)calloc(neval, sizeof(double));
  int ct = 0;
  trav = samples;
  while(trav->next != NULL) {
    int n = trav->n;
    int x = hamming_distance(best_samp[n], trav);
    int wx = weighted_hamming_distance(best_samp[n], trav);
    double y = percent_deviation(best_samp[n]->runtime, trav->runtime);
    
    F[ct] = trav->runtime;
    wD[ct] = wx;
    D[ct++] = x;
    //printf("x = %d, y = %0.4e\n", x, y);

    fprintf(fp_corr, "%d, %d, %f\n", x, wx, y);
    trav = trav->next;
  }
  double fdc = FDC(F, D, ct);
  double wfdc = FDC(F, wD, ct);
  printf("FDC = %f, wFDC = %f\n", fdc, wfdc);
  fprintf(fp_corr, "# FDC = %f, wFDC = %f\n", fdc, wfdc);

  free(F);
  free(D);
  free(wD);

  for (i = 0; i < max_dim + 1; ++i) {
    if (best_samp[i]) {
      fprintf(fp_corr, "# best = %s, %0.4e\n", int_to_string(best_samp[i]->pos, best_samp[i]->n), 
	      best_samp[i]->runtime);
    }
  }

  fclose(fp_corr);
  
  
  char *convex_name = get_program_name(ca);
  strcat(convex_name, "_convex.txt");
  FILE *fp_convex = fopen(convex_name, "w");
  if (fp_convex == NULL) {
    printf("Failed to open file\n");
    exit(1);
  }

  for (i = 0; i < max_dim + 1; ++i) {
    if (best_samp[i]) {
      int *convex = get_convex_diameter(samples, best_samp[i]);
      //printf("convex = %s\n", int_to_string(convex, i));      
      for (j = 0; j < i - 1; ++j) {
	fprintf(fp_convex, "%d, ", convex[j]);
      }
      fprintf(fp_convex, "%d\n", convex[i - 1]);
      free(convex);
    }
  }
  fclose(fp_convex);

  for (i = 0; i < max_dim + 1; ++i) {
    if (best_samp[i]) {
      printf("best = %s, %0.4e\n", int_to_string(best_samp[i]->pos, best_samp[i]->n), best_samp[i]->runtime);
    }
  }
  return 0;
}

// run the beam search best first search
int optimize_bfs(isl_ctx *ctx,  struct ppcg_options *options,
		 const char *input, const char *output, 
		 __isl_give isl_printer *(*print_cuda)(__isl_take isl_printer *p,
		 struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
	         struct gpu_types *types, void *user),struct coding_ants *ca, 
		 struct ca_node *regression_tree) {

  int r;

  best_first_search = 1;
  /* What level in the graph is the current 
   * beam search at.
   */
  beam_depth = 1;
  
  /* What level in the graph is the current ant at */
  current_depth = 0;

  /* What edge should be selected next */
  beam_edge = 0;

  struct ca_wrapper wrap; 
  wrap.o2_failed = 0;
  wrap.traverse_partition = traverse_partition_graph;
  wrap.traverse_schedule = traverse_schedule_graph;
  wrap.traverse_reduction = traverse_reduction_graph;
  wrap.traverse_texture = traverse_texture_graph;
  wrap.traverse_tile = traverse_tile_sizes;
  wrap.traverse_inner_tile = traverse_inner_tile;
  wrap.traverse_unroll = traverse_unroll_factors;
  wrap.traverse_atomic_store = traverse_atomic_store;
  wrap.traverse_cache_config = traverse_cache_config;
  wrap.traverse_shared_mem = traverse_shared_memory;
  wrap.optimization = 1; // O2 set schedule and partitioning 
  wrap.reducible = NULL;

  wrap.nread_only = 0;
  wrap.read_only = get_read_only_arrays(ca, &wrap.nread_only);
  ca->wrap = &wrap;

  // (1) initialize the ants
  struct ca_ant *ant = init_ants(ca, wrap.nread_only);

  ca->sched_graph = (struct ca_graph *)calloc(1, sizeof(struct ca_graph));
  ca->sched_graph->source = create_node(SOURCE, "source", NULL);

  beam_path = (struct path *)calloc(1, sizeof(struct path));
  beam_path->next = NULL;
  
  beam_node = ca->sched_graph->source;
  struct path *bp = beam_path;
  int num_eval = 0;
  ca->best_current_time = 100000;
  int n_one_edge = 0;  
  bfs_next_edge = -1;

  char *corr_name = get_program_name(ca);
  strcat(corr_name, "_bfs.txt");
  
  FILE *fp = fopen(corr_name, "w");
  if (fp == NULL) {
    printf("Failed to open file\n");
    exit(1);
  }

  for ( ; ; ) {
    beam_edge = bfs_next_edge + 1;   
    int best_edge;
    double best_time = 100000;  
    semantic_edge = -1;
    n_possible_edges = beam_node->n_edges;

    if (bfs_next_edge >= 0 && beam_edge >= beam_node->n_edges) {
      best_edge = bfs_next_edge;
      if (beam_node->out_edges[best_edge].semantic_edge) {
	semantic_edge = best_edge;
      }
      bfs_next_edge = -1;
      goto post_while;
    }

    do {
      current_depth = 0;
      current_path = beam_path;
      
      // traverse
      ca->wrap->node = ca->sched_graph->source;
      ca->wrap->ant = ant;
      ca->wrap->s_path = ant->s_path;
  
      // select a path through the graph
      ca_traverse(ant, ca);
      pre_ca_apply(ant, ca);

      if (regression_tree) {
	best_first_search = 0;
	traverse_regression_tree(ant, regression_tree, 1);
	ant->filter_set = 1;
	best_first_search = 1;
      }


      // generate the code
      r = generate_code(ctx, options, input, output, print_cuda, ca);
      
      printf("beam node = %s\n", beam_node->name);
      printf("lb = %s\n", int_to_string(ant->filter.attr_lb, num_regression_vec));
      printf("ub = %s\n", int_to_string(ant->filter.attr_ub, num_regression_vec));      
      printf("opt_vec = %s\n", int_to_string(ant->opt_vec, num_regression_vec));
      if (!satisfies_filter(ant)) {
	printf("best edge = %d\n", best_edge);
	post_ca_apply(ant, ca);   
	++beam_edge;
	continue;
      }
      assert (satisfies_filter(ant));
      
      if (semantic_edge >= 0) {
	post_ca_apply(ant, ca);      
	best_edge = semantic_edge;
	bfs_next_edge = -1;
	break;
      }
      
      if (n_possible_edges == 1) {
	printf("num next edges = 1\n");
	post_ca_apply(ant, ca);    
	best_edge = beam_edge;
	bfs_next_edge = current_bfs_next_edge;
	break;
      }
      
      double runtime = evaluate(ca, 0);

      post_ca_apply(ant, ca);

      if (runtime < best_time) {
	best_time = runtime;
	best_edge = beam_edge;
	bfs_next_edge = current_bfs_next_edge;
      }
      
      ++beam_edge;
      ++num_eval;
    } while(beam_edge < beam_node->n_edges);

  post_while:
    if (semantic_edge < 0) {
      bp->v = best_edge;
      bp->next = (struct path *)calloc(1, sizeof(struct path));
      bp->next->next = NULL;
      bp = bp->next;
      
      beam_depth++;
    }
    
    printf("best edge = %d, nedges = %d\n", best_edge, beam_node->n_edges);
    assert (best_edge >= 0 && best_edge < beam_node->n_edges);
    beam_node = beam_node->out_edges[best_edge].dst;

    if (beam_node->out_edges == NULL)
      break;

    printf("beam_depth = %d\n", beam_depth);
    printf("best edge = %d\n", best_edge);
    printf("current_beam node = %s\n", beam_node->name);
    printf("best so far = %0.4e\n", best_time);
    fprintf(fp, "%f\n", best_time);

    if (best_time < ca->best_current_time) {
      ca->best_current_time = best_time;
    }
  }

  fprintf(fp, "ppcg time = %f\n", ca->ppcg_only_time);
  fprintf(fp, "best time = %0.4e, num eval = %d\n", ca->best_current_time, num_eval);
  fclose(fp);

  return r;
}


// run the ant colony optimization alogorithm
struct ca_node *optimize(isl_ctx *ctx,  struct ppcg_options *options,
	       const char *input, const char *output, 
	       __isl_give isl_printer *(*print_cuda)(__isl_take isl_printer *p,
		struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
		struct gpu_types *types, void *user),struct coding_ants *ca) {

  int i, j, k;
  int r;

  double best_time = 100000;
  ca->best_current_time = best_time;
  struct ca_ant *best_ant;

  struct ca_wrapper wrap; 
  wrap.o2_failed = 0;
  wrap.traverse_partition = traverse_partition_graph;
  wrap.traverse_schedule = traverse_schedule_graph;
  wrap.traverse_reduction = traverse_reduction_graph;
  wrap.traverse_texture = traverse_texture_graph;
  wrap.traverse_tile = traverse_tile_sizes;
  wrap.traverse_inner_tile = traverse_inner_tile;
  wrap.traverse_unroll = traverse_unroll_factors;
  wrap.traverse_atomic_store = traverse_atomic_store;
  wrap.traverse_cache_config = traverse_cache_config;
  wrap.traverse_shared_mem = traverse_shared_memory;
  wrap.optimization = options->opt_level; // O2 set schedule and partitioning 
  wrap.reducible = NULL;

  wrap.nread_only = 0;
  wrap.read_only = get_read_only_arrays(ca, &wrap.nread_only);
  ca->wrap = &wrap;

  // (1) initialize the ants
  struct ca_ant *ants = init_ants(ca, wrap.nread_only);

  ca->sched_graph = (struct ca_graph *)calloc(1, sizeof(struct ca_graph));
  ca->sched_graph->source = create_node(SOURCE, "source", NULL);

  FILE *efp = fopen("experiments.txt", "w");
  if (efp == NULL) {
    printf("ERROR: experments fopen\n");
    exit(1);
  }
  
  int eval_num = 0;
  //wrap.node = ca->sched_graph->source;
  float last_metrics[6];
  memset(last_metrics, 0, 6*sizeof(float));
  
  double best_time_iter[NUM_ITER];
  struct ca_node *regression_tree = NULL;
  for (i = 0; i < NUM_ITER; ++i) {
    //current_iteration = i;
    int iteration_best = 0;
    double best_score_iter = 100000;
    if (kernel_change) {
      choose_best = 1;
    }

    kernel_change = 0;
    printf("Iteration %d\n", i);
    for (j = 0; j < NUM_ANTS; ++j) {      
      printf("Ant Number %d\n", j);
      if (ca->wrap->optimization == 2) {
	ca->wrap->o2_failed = 0;
      }
      
      // first ant chooses the best path so far
      // this handles the case where different ants may have
      // chosen optimal kernels but no ant chose
      // the combination of optimal kernels
      //choose_best = (i > 0 && j == 0);

      if (j > 0) {
	choose_best = 0;
      }

      ca->wrap->node = ca->sched_graph->source;
      ca->wrap->ant = &ants[j];
      ca->wrap->s_path = ants[j].s_path;

      // select a path through the graph
      ca_traverse(&ants[j], ca);
      
      pre_ca_apply(&ants[j], ca);
      if (j > 0 || i > 0) {
	printf("d1\n");
	ants[j].est_metrics[0] = last_metrics[0];
	ants[j].est_metrics[1] = last_metrics[1];
	ants[j].est_metrics[2] = last_metrics[2];
	ants[j].est_metrics[3] = last_metrics[3];
	ants[j].est_metrics[4] = last_metrics[4];
	ants[j].est_metrics[5] = last_metrics[5];
	printf("d2\n");
      }

      if (regression_tree && !choose_best) {
	if (j < NUM_ANTS / 2) {
	  printf("d3\n");
	  traverse_regression_tree(&ants[j], regression_tree, 1);
	  printf("d4\n");
	}
	else {
	  if (j == NUM_ANTS / 2) {
	    assert (regression_tree != NULL);
	    printf("d5\n");
	    reset_regression_tau(regression_tree);
	    printf("d6\n");
	    deposit_reg_tree_all(regression_tree, i * NUM_ANTS, ca->best_current_time);
	    printf("d7\n");
	  }
	  printf("d8\n");
	  traverse_regression_tree(&ants[j], regression_tree, 0);
	  printf("d9\n");
	}
	//traverse_regression_tree(&ants[j], regression_tree, (j < NUM_ANTS/2));
	ants[j].filter_set = 1;
      }

      // apply the optimizations
      ca_apply_optimizations(ca, &ants[j], options); // TODO: deprecate
      
      // generate the code
      r = generate_code(ctx, options, input, output, print_cuda, ca);

      if (ca->wrap->optimization == 2) {
	if (ca->wrap->o2_failed) {
	  ants[j].score = 1e9;
	  post_ca_apply(&ants[j], ca);
	  samples[i * NUM_ANTS + j].attr = NULL;
	  samples[i * NUM_ANTS + j].score = -1;
	  
	  continue;
	}
      }

      printf("lb = %s\n", int_to_string(ants[j].filter.attr_lb, num_regression_vec));
      printf("ub = %s\n", int_to_string(ants[j].filter.attr_ub, num_regression_vec));
      printf("opt_vec = %s\n", int_to_string(ants[j].opt_vec, num_regression_vec));
      //if (wrap.optimization < 2 && !choose_best)
      //	assert (satisfies_filter(&ants[j]));
      // reporting 
      for (k = 0; k < ants[j].ntexs; ++k) {
	printf("texture %s\n", ants[j].texs[k]);
      }
      printf("tile sizes = %s\n", ants[j].sizes);
      printf("unroll = %s\n", ants[j].unroll_factors);

      printf("est metrics = %s\n", float_to_string(ants[j].est_metrics, 6));
      printf("iteration %d, ant %d, choose best %d\n", i, j, choose_best);
      
      if (ca->wrap->optimization == 0) {
	printf("\nalg = %s\n", (ants[j].compiler_options[CO_ALG] == 0 ? "ISL" : "FEAUTRIER"));
	printf("serialize scc = %d\n", ants[j].compiler_options[CO_SERIALIZE]);
	printf("max depth = %d\n", ants[j].compiler_options[CO_MAX_DEPTH]);
	printf("outer coincidence = %d\n", ants[j].compiler_options[CO_OUTER_COIN]);    
      }

      double runtime = evaluate(ca, 1);
      
      samples[i * NUM_ANTS + j].attr = (int *)calloc(num_regression_vec, sizeof(int));
      memcpy(samples[i * NUM_ANTS + j].attr, 
	     ants[j].opt_vec, 
	     num_regression_vec * sizeof(int));
      samples[i * NUM_ANTS + j].score = runtime;
      memset(samples[i * NUM_ANTS + j].metric, 0, 6 * sizeof(double));
      if (wrap.optimization == 2) {
	for (k = 0; k < ca->n_stmts; ++k) {
	  assert (samples[i * NUM_ANTS + j].attr[k + skewing_start] == 0 ||
		  samples[i * NUM_ANTS + j].attr[k + skewing_start] == 1);
	}
      }
      ants[j].score = runtime;
      int gathered = read_metrics_csv(&ants[j]);
      
      if (gathered) {
	int k, l;
	unsigned long sum_cycles = 0;
	for (k = 0; k < ants[j].nkernel; ++k) {
	  sum_cycles += ants[j].kernel_cycles[k];
	}
       
	for (l = 0; l < 6; ++l) {
	  int rel = 0;
	  double metric = 0;
	  for (k = 0; k < ants[j].nkernel; ++k) {	    
	    double weight = (sum_cycles > 0 ? (double) ants[j].kernel_cycles[k] / sum_cycles : 0.01);
	    if (ants[j].kernel_metrics[k][l] > 0) {
	      //last_metrics[l] += ants[j].kernel_metrics[k][l] * weight;
	      metric += ants[j].kernel_metrics[k][l] * weight;
	      ++rel;
	    }
	  }

	  if (runtime < best_time)
	    last_metrics[l] = metric;

	  samples[i * NUM_ANTS + j].metric[l] = metric;
	} // end for l      
	printf("metrics = %s\n", double_to_string(samples[i * NUM_ANTS + j].metric, 6));
      }

      // update ants path with metrics info
      update_metrics(&ants[j], ca->sched_graph->source, ants[j].s_path, (runtime < best_time));
      if (runtime < best_time) {
	best_time = runtime;
	best_ant = copy_ant(ants[j]);

	save_current_best_schedule(ca);
	//for (k = 0; k < ca->edges->max_dim; ++k) {
	//  best_ant.sccs[k] = int_copy(ants[j].sccs[k], ca->n_stmts);
	//}
      }

      fprintf(efp, "%d, %f\n", (eval_num++), best_time);

      if (runtime < best_score_iter) {
	best_score_iter = runtime;
	iteration_best = j;
      }

      printf("num reg %d\n", num_regression_vec);
      printf("ant %d opt_vec = %s\n", j, int_to_string(ants[j].opt_vec, ants[j].opt_vec_ptr));
      //exit(0);
      post_ca_apply(&ants[j], ca);
    } // end for j

    ca->wrap->node = ca->sched_graph->source;

    if (best_time < ca->best_current_time) {
      ca->best_current_time = best_time;
    }

    best_time_iter[i] = ca->best_current_time;
    
    if (i >= 2) 
      update_graph(ants, ca, *best_ant);
    else
      update_graph(ants, ca, ants[iteration_best]);
    
    struct ca_node *old_tree = regression_tree;
    regression_tree = build_regression_tree(best_ant, (i + 1) * NUM_ANTS);
    if (!regression_tree) {
      // than use the old tree
      regression_tree = old_tree;
    }
    update_regression_tree_edges(regression_tree, best_ant, (i + 1) * NUM_ANTS);

    printf("-----------------------------\n");
  } // end num iter
  
  //printf("running best ant\n");
  // run the best ant
  //pre_ca_apply(best_ant, ca);
  //ca_apply_optimizations(ca, best_ant, options);

  // generate the code
  //r = generate_code(ctx, options, input, output, print_cuda, ca);
  
  //evaluate(ca);
  fprintf(efp, "ppcg time = %f\n", ca->ppcg_only_time);
  fclose(efp);

  for (i = 0; i < NUM_ITER; ++i) {
    printf("iteration %d best time = %0.4e\n", i, best_time_iter[i]);
  }

  printf("Best Time = %0.4e\n", best_time);
  printf("Best Ant\n");
  for (i = 0; i < best_ant->ntexs; ++i) {
    printf("texture %s\n", best_ant->texs[i]);
  }
  printf(" tile sizes = %s\n", best_ant->sizes);
  printf(" unroll = %s\n", best_ant->unroll_factors);
 
  printf("graphSize = %d\n", graphSize);
  return regression_tree;
}

/* We assume that all paths have the same depth */
void set_depth_var(struct ca_graph *graph) {
  int depth = 0;
  struct ca_node *node = graph->source;//->out_edges[0].dst;

  while (node->out_edges != NULL) {
    node = node->out_edges[0].dst;
    ++depth;
  }

  graph->depth = depth;
}


isl_stat get_name_from_set(__isl_take isl_set *set, void *user) {
  struct name_helper *data = (struct name_helper *)user;
  data->arr[data->idx++] = strdup(isl_set_get_tuple_name(set));
  return isl_stat_ok;
}

isl_bool get_leaf_name(__isl_keep isl_schedule_node *node, void *user) {
  int i;

  if (isl_schedule_node_get_type(node) == isl_schedule_node_leaf) {
    isl_union_set *dom = isl_schedule_node_get_domain(node);
    //isl_schedule_node_dump(node);
    //isl_union_set_dump(dom);
    //assert(isl_union_set_n_set(dom) == 1);

    struct name_helper *data = (struct name_helper *)user;
    
    isl_union_set_foreach_set(dom, &get_name_from_set, user);
    //isl_set *stmt = isl_set_from_union_set(dom);
    //data->arr[data->idx++] = strdup(isl_set_get_tuple_name(stmt));
    return isl_bool_true;
  }
  
  
  return isl_bool_true;
}

char **get_schedule_stmt_array(isl_schedule_node *node, int n, int *size) {
  char **arr = (char **) calloc(n, sizeof(char *));

  struct name_helper helper;
  helper.arr = arr;
  helper.idx = 0;

  isl_schedule_node_foreach_descendant_top_down(node, &get_leaf_name, &helper);

  *size = helper.idx;
  return arr;
}




// return the number of values < 0
int num_remaining(int *vec, const int len) {
  int i = 0;
  int ct = 0;
  for (i = 0; i < len; ++i) {
    if (vec[i] < 0) {
      ++ct;
    }
  }

  return ct;
}

int is_legal(struct simple_edges *edges, int *sccs, const int len) {
  int i;
  for (i = 0; i < edges->n_edge; ++i) {
    if (sccs[edges->src[i]] < 0 || sccs[edges->dst[i]] < 0) {
      continue;
    }
    if (sccs[edges->src[i]] > sccs[edges->dst[i]]) {
      return 0;
    }
  }

  return 1;
}

// Adds children to node for each possilbe partition of 'size' stmts
// to 'scc_id'. start and end denote the range of stmts in which it is
// legal to place in the current scc.
void part_size_populate(struct ca_node *node, 
			struct simple_edges *edges, 
			int *sccs_vec, const int len, 
			int start, int end, int size, 
			int *e_idx, int scc_id, const int n_part) {
  assert(node->out_edges != NULL);

  int i;
  int *sccs = int_copy(sccs_vec, len);

  //printf("size = %d, start = %d, end = %d\n", size, start, end);
  
  if (size == 0) {
    //printf("sccs = %s\n", int_to_string(sccs, len));
    if (!is_legal(edges, sccs, len)) {
      node->n_edges--;
      free(sccs);
      return;
    }

    node->out_edges[(*e_idx)++].dst = create_node(PART, int_to_string(sccs, len), sccs);
    int remaining_sccs = num_remaining(sccs, len);
    if (remaining_sccs > 0) {
      int status = part_size_graph(node->out_edges[*e_idx-1].dst, edges, len, 
				   n_part - 1, remaining_sccs, scc_id +1);
      
      if (status < 0) {
	// dst has no children
	--(*e_idx);

	free(node->out_edges[*e_idx].dst);
	node->out_edges[*e_idx].dst = NULL;
	--node->n_edges;	
      }
    }
    return;    
  }

  if(start == end) {
    return;
  }
    
  // ca_allocate_out_edges(node, end - start);
  for (i = start; i < end; ++i) {    
    if (sccs[i] >= 0) {
      // already placed
      continue; 
    }
    
    //printf("Begin = %s: %d\n", int_to_string(sccs, len), i);

    sccs[i] = scc_id;

    //struct ca_node *child = create_node(PART, int_to_string(sccs, len), 
    //					int_copy(sccs, len));

    part_size_populate(node, edges, sccs, len, i + 1, end, size - 1, e_idx, 
		       scc_id, n_part);
    
    if (node->n_edges <= 0) {
      break;
    }
    //printf("End %s\n", int_to_string(sccs, len));
    sccs[i] = -1;
  }
  free(sccs);
}


// n choose k
long choose(long n, long k) {
  int i;

  if (n <= k) {
    return 1;
  }

  long num = 1;
  long den = 1;
  for (i = n - k + 1; i <= n; ++i) {
    
    num *= i;
  }

  for (i = 1; i <= k; ++i) {
    den *= i;
  }

  return (num / den);
}

// create a node for each number of statements in a partition
int part_size_graph(struct ca_node *node, struct simple_edges *edges, 
		     const int len, int n_part, int n_stmts, int scc) {
  assert (n_part > 0);
  //printf("number of partitions = %d\n", n_part);
  int i;

  int *sccs = NULL;
  if (node->type == PART) {
    sccs = (int *)node->val;
  }
  else { // init
    sccs = (int *)calloc(n_stmts, sizeof(int));
    for (i = 0; i < n_stmts; ++i) {
      sccs[i] = -1;
    }    
  }

  if (n_part == 1) {
    // base case
    ca_allocate_out_edges(node, 1);
    
    char buf[50];
    snprintf(buf, 50, "part_size:%d\n", n_stmts);

    int *v = (int *)calloc(1, sizeof(int));
    *v = n_stmts;
    
    //node->out_edges[0].dst = create_node(PART_SIZE, buf, v);
    struct ca_node *child = create_node(PART_SIZE, buf, v);
    ca_allocate_out_edges(child, 1);

    int e_idx = 0;

    part_size_populate(child, edges, sccs, len, 0, len, n_stmts, &e_idx, scc, n_part);
    if (e_idx == 0) {
      free(child);
      return -1;
    }

    assert(e_idx > 0);
    node->out_edges[0].dst = child;    
    return 0;
  }

  // number of statements in the partition
  const int max_size = n_stmts - n_part + 1;
  const int min_size = 1;     
  
  ca_allocate_out_edges(node, max_size);

  for (i = min_size; i <= max_size; ++i) {
    char buf[50];
    snprintf(buf, 50, "part_size:%d", i);
    printf("\nname = %s\n", buf);
    int *v = (int *)calloc(1, sizeof(int));
    *v = i;

    int next_part = n_part - 1;
    int next_stmts = n_stmts - i;
    
    struct ca_node *child = create_node(PART_SIZE, buf, v);
    ca_allocate_out_edges(child, choose(n_stmts, i));
    int cedges = child->n_edges;

    int e_idx = 0;
    part_size_populate(child, edges, sccs, len, 0, len, i, &e_idx, scc, n_part);

    if (e_idx == 0) {
      // no legal partition
      free(child);
      --node->n_edges;
      continue;
    }

    assert (e_idx > 0);
    child->n_edges = e_idx;

    node->out_edges[i-1].dst = child;
  }
  
  if (node->n_edges <= 0) {
    return -1;
  }

  return 0;
}

struct ca_node *build_graph_from_node(isl_schedule_node *node, const int n_stmts,
			   struct simple_edges *scc_graph, struct scc_vec *vec) {

  struct ca_node *source;
  // the number of sccs contained in node
  int n_scc = 0;

  // Get the scc ids contained in node
  int *node_sccs = get_sccs_from_stmts(scc_graph, n_stmts, node, &n_scc);

  vec->sccs = node_sccs;
  vec->len = n_scc;

  source = create_node(SOURCE, "source_scc_order", vec);  
  
  int n_valid = get_num_valid_sccs(scc_graph, node_sccs, n_scc);
  int i;
  for (i = 0; i < scc_graph->n_scc; ++i) {
    printf("scc %d, n_unsat = %d, n_in = %d\n", scc_graph->node[i].scc, 
	   scc_graph->node[i].n_unsat, scc_graph->node[i].n_in);

  }
  assert(n_valid > 0);  

  ca_allocate_out_edges(source, n_valid);  
  return source;
}

void build_root(isl_schedule_node *node, struct ca_graph *graphs, 
		struct simple_edges *scc_graph, const int n_part, const int n_stmts) {
  
  int i, j;

  struct scc_vec *vec = (struct scc_vec *)calloc(n_part, sizeof(struct scc_vec));
  for (i = 0; i < n_part; ++i) {
    node = isl_schedule_node_child(node, i);
      
    graphs[i].source = build_graph_from_node(node, n_stmts, scc_graph, &vec[i]);

    // update the scc graph so we know what are valid starting
    // sccs in the next partition
    for (j = 0; j < vec[i].len; ++j) {
      update_scc_edges(scc_graph, vec[i].sccs[j]);
    }

    node = isl_schedule_node_parent(node);
    // number of stmts + n_part + n_part_size
    graphs[i].depth = n_stmts + 1 + n_stmts + 1;
  }  
}

void build_sched_root(struct coding_ants *ca, isl_schedule_node *node) {
  int i, j;
  int n;
  int n_part;
  int len = 0;
  struct simple_edges *scc_graph = ca->edges;

  // sanity check
  for (i = 0; i < scc_graph->n_scc; ++i) {
    assert (scc_graph->node[i].scc == i);
  }

  isl_union_set *dom;
  dom = isl_schedule_node_domain_get_domain(node);

  // n stmts
  n = isl_union_set_n_set(dom);
  ca->n_stmts = n;  

  assert (n > 0);

  node = isl_schedule_node_child(node, 0);
  if (isl_schedule_node_get_type(node) == isl_schedule_node_set) {
    // set nodes could be fused as they do not imply an ordering
    node = isl_schedule_node_parent(node);
    n_part = 1;
  }
  else {  
    // the number of partitions (i.e. loop nests)
    n_part = isl_schedule_node_n_children(node);
  }

  // for now
  //assert (n_part == 1);
 
  if (n_part == 1) {
    // node = isl_schedule_node_parent(node);
  }
  
  //printf("n bands == %d\n", isl_schedule_node_band_n_member(node));
  //exit(0);

  ca->sched_graph = (struct ca_graph *)calloc(n_part, sizeof(struct ca_graph));
  ca->n_part = n_part;
  
  reset_graph(scc_graph);
  build_root(node, ca->sched_graph, scc_graph, n_part, n);
  reset_graph(scc_graph);
}


void set_schedule_graph(struct coding_ants *ca) {
  int i, j;
  int n, n_part;

  isl_schedule *schedule;
  isl_schedule_node *node;
  isl_union_set *dom;

  char sched_file[50];
  snprintf(sched_file, 50, "%s_ref.sched", get_program_name(ca));

  schedule = load_schedule(ca->ctx, sched_file);
  node = isl_schedule_get_root(schedule);
  dom = isl_schedule_node_domain_get_domain(node);

  build_sched_root(ca, node);
}

void init_graph(struct coding_ants *ca) {
  int i, j, k; // loop iterators
  printf("init_graph\n");
  // initialize random number generator
  srand(0);//time(NULL));

  node_table = NULL;
  kernel_best_time_table = NULL;
  //node_table = (struct tuple*)calloc(1, sizeof(struct tuple));
  //node_table->next = NULL;

  // set the compiler options graph
  ca->compiler_options_graph.source = create_node(SOURCE, "source", NULL);
  set_shared_mem_graph(ca->compiler_options_graph.source);

  // set the memory options graph
  ca->mem_opt_graph.source = create_node(SOURCE, "source", NULL);
  //set_mem_opt_graph(ca->mem_opt_graph.source, ca);

  // (4) set the depth for the compiler options graph
  set_depth_var(&ca->compiler_options_graph);
  set_depth_var(&ca->mem_opt_graph);
  
  samples = (struct regression_sample *)calloc(NUM_ANTS * NUM_ITER, 
					       sizeof(struct regression_sample));

  /*set_schedule_graph(ca);

  int n_kernels = ca->n_kernels;
  // allocate a graph for each kernel
  ca->graph = (struct ca_graph*)calloc(n_kernels, sizeof(struct ca_graph));

  for (i = 0; i < n_kernels; ++i) {
    // set the kernel_name for this graph
    ca->graph[i].kernel_name  = (char *)calloc(50, sizeof(char));
    snprintf(ca->graph[i].kernel_name, 50, "kernel%d", i);

    // (1) allocate the source nodes
    ca->graph[i].source =  create_node(SOURCE, "source", NULL);

    const int tile_dim = ca->n_tile_dims[i];
    const int block_dim = ca->n_block_dims[i];   

    // (3) set the cuda dimensions
    set_cuda_dims(ca->graph[i].source, i, tile_dim, block_dim);

    // (4) set the maximum depth
    set_depth_var(&ca->graph[i]);

    #ifdef DEBUG
    for (j = 0; j < ca->graph[i].source->n_edges; ++j) {
      printf("%s\n", (char *)ca->graph[i].source->out_edges[j]->val);   
      for (k = 0; k < ca->graph[i].source->out_edges[j]->n_edges; ++k) {
	printf("\t%s\n", (char *)ca->graph[i].source->out_edges[j].out_edges[k]->val);
      }
    }
    #endif
  } // end for i
  */
  #ifdef DEBUG
  for (i = 0; i < ca->compiler_options_graph.source->n_edges; ++i) {
    printf("%d\n", *((int *)ca->compiler_options_graph.source->out_edges[i]->val));
  }
  #endif
}
