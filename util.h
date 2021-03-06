#ifndef UTIL_H
#define UTIL_H

#include <string.h>

#include <isl/space.h>
#include <isl/val.h>

/* Compare the prefix of "s" to "prefix" up to the length of "prefix".
 */
static inline int prefixcmp(const char *s, const char *prefix)
{
	return strncmp(s, prefix, strlen(prefix));
}

__isl_give isl_multi_val *ppcg_multi_val_from_int(__isl_take isl_space *space,
	int val);

#endif
