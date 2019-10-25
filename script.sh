python3 create_mnist.py 
python3 train_mnist.py --epochs=100 --ckpt-dir='./model_100_epochs' --summary-dir='./summary_100_epochs'
python3 train_mnist.py --epochs=100 --ckpt-dir='./model_100_epochs_lr_2e-5' --summary-dir='./summary_100_epochs_2e-5'