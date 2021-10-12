import os

num_epochs = 5000

for epoch in range(num_epochs):
  os.system("python3 inpaint.py")
#  os.system("python3 global_gan.py")
  print('Epoch : %d/%d' % (epoch+1, num_epochs))

  os.system("python3 edge.py")
  print('Epoch : %d/%d' % (epoch+1, num_epochs))  
