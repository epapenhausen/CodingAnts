#ifndef _CA_H
#define _CA_H

#define SOURCE 0
#define COMPILER_OPTION 1
#define CUDA_CONFIG 2 
#define MEM_PLACE 3
#define NUM_PARTS 4
#define PART_SIZE 5
#define PART 6
#define SCC 7
#define SCHED 8
#define SCHED_TYPE 9
#define TOGGLE_RED 10
#define MEM_PLACE_SOURCE 11
#define UNROLL_SOURCE 12
#define UNROLL 13
#define ATOMIC_SOURCE 14
#define ATOMIC 15
#define CACHE_SOURCE 16
#define CACHE 17
#define REGRESSION 18
#define SHARED 19

#define RHO 0.1
#define CA_Q 0.15

#define NUM_ANTS 10
#define NUM_ITER 5

#define MAX_REGS 64
#define MAX_BANDWIDTH 86.4
#define ALPHA 1//0.66
#define BETA 1//5//0.66//34

#define CO_ALG 0
#define CO_SERIALIZE 1
#define CO_MAX_DEPTH 2
#define CO_OUTER_COIN 3

struct scc_node;

struct scc_edge {
  /* Source and destination scc nodes */
  struct scc_node *src;
  struct scc_node *dst;
  
  /* Pointer to the next edge in the list */
  struct scc_edge *next;
};

/* Helper for filtering potential paths based 
 * on selected optimizations in the regression tree
 */
struct regression_filter {
  /* Array of length num_regression_vec, which contains split values
   * -1 indicates no split on the attribute has been selected
   */
  //int *attr_vec; // TODO: NEED LOWER AND UPPER BOUND
  int *attr_lb;
  int *attr_ub;

  /* Mask of length num_regression_vec, which specifices whether the
   * split is <= (i.e. 0) or > (i.e. 1)
   */
  //int *lte_mask;
};

struct regression_sample {
  int *attr;
  double score;
  double metric[6];
};

struct scc_node {
  /* Array of statement ids belonging to this scc */
  int *stmts;
  
  /* Stmt spaces */
  char **spaces;
  
  /* The number of statements in this scc */
  int n;

  /* SCC number */
  int scc;

  /* A linked list of outgoing edges */
  struct scc_edge *edges;

  /* Number of incoming edges */
  int n_in;

  /* The number of unsatisfied incoming dependence edges */
  int n_unsat;
};

// simple edge representing a source 
// and destination statement index
struct simple_edges {
  /* Array of src indices, one for each edge. (i.e. edge i's src is at src[i]) */
  int *src;

  /* Array of dst indices, one for each edge. (i.e. edge i's dst is at dst[i]) */
  int *dst;
  
  /* Number of edges */
  int n_edge;

  /* Array of scc nodes in the scc graph */
  struct scc_node *node;
  
  /* Number of scc nodes */
  int n_scc;

  /* Number of statements */
  int n_stmts;
  
  char** stmt_str;

  /* An array of stmt strings (e.g. S_0) to map stmts to integer ids */
  char **domain_stmt_array;

  /* The maximum dimension of all stmts */
  int max_dim;

  /* the current (maximal) number of linearly independent rows in the node schedules*/
  int n_row;

  int partition_start;
  
  int *stmt_ids;
};

struct ca_node {
  /* Optimization type (e.g. block size, toggle shared mem, etc.) */
  int type;
  
  char *name;
  
  /* Value of optimization (e.g. block=32, shared mem=true, etc) */
  void *val;

  /* Optional auxilary data */
  void *aux;

  /* There may be cycles. This helps prevent infinite loops*/
  int visited;

  /* Optional subgraph */
  struct ca_graph *subgraph;

  /* Number of optional subgraphs */
  int n_subgraph;

  /* Number of outgoing edges */
  int n_edges;
  
  /* Number of implicit outgoing edges */
  int n_implicit;

  /* List of outgoing edges */
  struct ca_edge *out_edges;

  /* A single edge with null dst that represents a 
   * group of edges that have not yet been traversed 
   * and so they do not exist yet. */
  struct ca_edge *implicit_edge;

  /* CUDA Kernel this node corresponds to. 
   * May be -1 to indicate no specific kernel.
   */
  int kernel_id;

  /* List of stmts that are being optimzed
   * with this node. Useful for inferring 
   * the kernel id.
   */
  char **stmt_str;
  int stmt_len;

  /* True once we've seen a coincident node */
  int in_kernel;

  /* best kernel score */
  unsigned long kernel_score;

  double best_score;
};

struct ca_edge {
  int has_been_traversed;

  /* Does this edge need to be taken */
  int semantic_edge;
  
  /* Index of the metrics of the best ant in 'metrics' */
  int best_metrics_id;

  /* Unique identifier */
  int edge_id;
  
  /* Pheremone value on the edge */ 
  double tau;

  /* Pheremone value at the start of the iteration */
  double tau_start;

  /* Array of metric values for each ant that has 
   * crossed this edge (useful for stddev)
   */
  float **metrics;
  int metric_ptr;
  
  /* Source of the edge */
  struct ca_node *src;
  
  /* Destination of the edge */
  struct ca_node *dst;
};

struct ca_array {
  /* Name of the array */
  char *name;

  /* Is the array read-only? */
  int read_only;
};

struct ca_graph {
  /* A unique identifier for this graph */
  char *kernel_name;
  
  struct ca_node *source;
  struct ca_node *sink;  

  /* Depth of the graph */
  int depth;

  /* Is the graph not fully built */
  int implicit;
};

struct path {
  /* The edge number taken */
  int v;

  /* A pointer to the next path struct */
  struct path *next;
};

struct ca_ant {
  
  int do_reduction;

  /* Vector describing optimizations to be processed by the
   * regression tree.
   */
  int *opt_vec;
  int opt_vec_ptr;

  struct regression_filter filter;
  int filter_set;
  
  /* Flag for testing if the ant has processed the 
   * first partition yet.
   */
  int first_partition;

  /* Array of compiler optimizations selected */
  struct ca_node compiler_opts[1];
    
  /* Array of edge indices taken by this ant */
  int compiler_path[1];

  int *compiler_options;
  
  /* String of tile sizes for reporting */
  char *sizes;

  /* String of loop unroll factors for reporting */
  char *unroll_factors;

  /* Arrays placed in texture memory */
  int ntexs;
  char **texs;

  /* Array of optimization nodes selected for memory placement */
  struct ca_node *mem_opts;
  
  /* Array of edge indices taken for each sched graph */
  struct path **sched_path;
  //int **sched_path;

  struct path *s_path;
  
  /* Array of edge indices taken in the mem graph */
  int *mem_opt_path;
  
  /* Array of optimization nodes selected for each kernel */
  struct ca_node **opts;

  /* Array of edge indices taken by this ant for each kernel */
  int **opt_path;

  /* Depth of the graph for each kernel */
  int *n_opts;

  /* Partition of sccs from schedule graph*/
  int **sccs;

  /* Number of kernels in this ant's implementation */
  int nkernel;
  
  /* Array of estimated metric values */
  float *est_metrics;
  /* Number of times a value has been estimated. 
   * Useful for computing running averages.
   */
  int n_est; 

  /* Array of nvprof metrics for each kernel */
  float **kernel_metrics;

  /* Array of stmt strings for each kernel */
  char ***kernel_stmts;
  
  /* The number of stmts per kernel */
  int *kernel_stmts_len;
  
  /* Current kernel id */
  int kernel_stmt_id;

  unsigned long *kernel_cycles;  
  
  /* Score of selected optimizations based on runtime */
  double score;
};

struct ca_config {
  /* List of array names to be placed in texture memory */
  char **tex_array;
  
  /* Number of arrays to be placed in texture memory */
  int n_tex_array;

  /* Partition of sccs selected by an ant */
  int **sccs;
};

/* Struct to be passed to the isl scheduler. 
 * Traversal will occur as the scheduling progresses.
 */
struct ca_wrapper {
  int o2_failed;

  int *compiler_options;

  /* Optimiztion level to set how much 
   * control is given to coding ants.
   */
  int optimization;

  /* The current traversing ant */
  struct ca_ant *ant;

  /* Current node the ant is on */
  struct ca_node *node;

  /* Current pointer to the path list. It may not be
   * at the head of the list.
   */
  struct path *s_path;

  /* List of names of read-only arrays */  
  int nread_only;
  struct ca_array *read_only;
  
  /* Set of reducible statements */
  isl_union_set *reducible;

  /* Function pointer to traverse the graph */
  int *(*traverse_partition)(struct ca_ant *, struct path *, struct ca_node *, 
			     struct simple_edges *, const int, const int);

  /* Function pointer to traverse the graph */
  isl_vec *(*traverse_schedule)(struct ca_ant *, struct path *, struct ca_node *, struct simple_edges *, isl_mat *, const int);

  /* Function to select reduction nodes */
  int (*traverse_reduction)(struct ca_ant *, struct path *, struct ca_node *, struct simple_edges *);

  /* Function to select texture arrays */
  char **(*traverse_texture)(struct ca_ant *, struct path *, struct ca_node *, struct ca_array *, int, int *);

  /* Function to select outer tile and block sizes */
  int **(*traverse_tile)(struct ca_ant *, struct path *, struct ca_node *, int *, int, char *,
			char **, const int);

  /* Function to select inner tile sizes */
  int *(*traverse_inner_tile)(struct ca_ant *, struct path *, struct ca_node *, 
			      const int, int*, char*, char**, const int);

  /* Function to select unroll factors */
  //int (*traverse_unroll)(struct ca_ant *, struct path *, struct ca_node *, char **, const int);
  int *(*traverse_unroll)(struct ca_ant *, struct path *, struct ca_node *, 
			  const int, const int, char **, const int);

  /* Function to select whether we atomic store or not */
  int (*traverse_atomic_store)(struct ca_ant *, struct path *, struct ca_node *, char *);

  /* Function to select the cache configuration */
  int (*traverse_cache_config)(struct ca_ant *, struct path *, struct ca_node *, const int);

  /* Function to toggle shared memory usage */
  int (*traverse_shared_mem)(struct ca_ant *, struct path *, struct ca_node *, char *);
 
};

/* Coding ants struct to store relevant information needed
 * to construct the optimization graph and perform the 
 * optimization.
 */
struct coding_ants {
  /* Name of the input c file */
  char *input;

  /* Initialization flag */
  int init;

  /* The number of non-scalar arrays */
  int n_array;
  
  /* The number of statements */
  int n_stmts;

  /* List of non-scalar arrays */
  struct ca_array *arrays;

  /* Number of GPU kernels */
  int n_kernels;

  /* Number of tile dimensions per kernel */
  int *n_tile_dims;

  /* Number of grid dimensions per kernel */
  int *n_grid_dims;

  /* Number of block dimensions per kernel */
  int *n_block_dims;

  /* Optimization graph per kernel*/
  struct ca_graph *graph;

  /* Optimization graph for compiler options */
  struct ca_graph compiler_options_graph;

  /* Optimization graph for memory placement */
  struct ca_graph mem_opt_graph;

  /* Optimization graph for the scheduler */
  struct ca_graph *sched_graph;

  /* The number of partitions or (sched_graphs) */
  int n_part;

  /* Options to be passed to ppcg */
  struct ca_config config;

  /* A list of legality dependence edges. */
  struct simple_edges *edges;

  struct ca_wrapper *wrap;

  isl_ctx *ctx;

  double orig_run_time;
  
  double best_current_time;
  
  double ppcg_only_time;
};

struct name_helper {
  char **arr;
  int idx;
};

void test();

//isl_stat get_set_name(isl_set *set, void *user);
//char **get_stmt_array(isl_union_set *dom);

struct coding_ants *coding_ants_alloc(const char* input);
void set_schedule_graph(struct coding_ants *ca);
int exists(char *string, char **array, int n_array);
void init_graph(struct coding_ants *ca);
void compile(struct coding_ants *ca, const char* type);
double get_run_time(struct coding_ants *ca, const char* type);
void ca_traverse_partition_level();
#endif
