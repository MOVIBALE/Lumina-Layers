import os
print('Current directory:', os.getcwd())
with open('testfile.txt', 'w') as f:
    f.write('HELLO WORLD')
print('File created successfully')
print('File exists:', os.path.exists('testfile.txt'))
