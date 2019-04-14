import os
file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

musiclabel = list()
for cls in classes:
    music = input('Enter the label of {}:'.format(cls))
    musiclabel.append(music)

with open('musiclabel.txt', 'w') as f:
    for item in musiclabel:
        f.write("%s\n" % item)
