import csv

a = open("digit_5.csv", "r+")
b = a.readlines()
a.close()

grid = [b[i:i+28] for i in range(0, len(b), 28)]


with open('grid.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(grid)