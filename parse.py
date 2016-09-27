import itertools
import nltk
import re

def save(lines):
    with open('C:/arr.txt', 'w') as myfile:
        myfile.write('\n'.join(lines))

print "Reading CSV file..."
with open('C:/part1.txt', 'rb') as f:
    text = str(f.read())
    print 'delete starange symbols'
    res = re.split('\n|\t|\r|{|}|\[|\]|"data"|"elements"|"link"|name=|"| : |'+
                   '<div>|<a>|<a/>|<li>|<li/>|class=|title|src=|<a|'+
                   '<a href=|href=|<div|<|>', text)
    res = filter(lambda x: len(x)>3, res)
    save(res)
   




