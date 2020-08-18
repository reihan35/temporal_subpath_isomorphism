import matplotlib.pyplot as pyplot


def make_diagram(values,errorValues):
    pyplot.bar(range(len(values)), values, color = 'skyblue')
    pyplot.errorbar(range(len(values)), values, yerr = errorValues,
        fmt = 'none', capsize = 10, ecolor = 'red', elinewidth = 2, capthick = 8)
    pyplot.show()

values = [5, 3, 7, 9]
errorValues = [1, 0.5, 2.5, 3]
make_diagram(values,errorValues)