import imp
Sort = imp.load_source('sorting', './lib/sorting.py').Sort

sort = Sort(num_of_digits=4, verbose=False)
sort.sarsa_lambda()
sort.print_qvalues()

print "Save obj to salsa_lambda.dat"
Sort.save_to_file(sort, "salsa_lambda.dat")
