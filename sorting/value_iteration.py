import imp
Sort = imp.load_source('sorting', './lib/sorting.py').Sort

sort = Sort(num_of_digits=4)
sort.value_iteration()
sort.print_num_of_swap()

print "Save obj to value_iteration.dat"
Sort.save_to_file(sort, filename = "value_iteration.dat")
