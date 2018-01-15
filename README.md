# Machine-Learning-tensorflow
Course work/course design in CMPS242, Autumn 2017, UCSC

<h2>1. Ham or Spam</h2>

- Text Classification problem
- Used NLTK(Natural Language Toolkit) to extract text feature
- Implemented Batch/Stochastic Gradient Descent logistic regression
- Implemented EG+-(Exponentiated Gradient +-). Reference: <a href='http://hunch.net/?p=286'>Exponentiated Gradient +-</a>
- Compared among different algorithms
- Achieved 99% Correctness

<h2>2. Trump VS Clinton</h2>

- Complex Text Classification problem
- Implemented LSTM(Long Short-term memory) RNN(Recurrent Neural Network) with Feed-Forward Neural Network
- Cross Entropy Soft Max Loss Function
- Achieved 90% Correctness

<h2>3. Image Captioning</h2>

- Used Flickr-8k dataset
- Used Keras-tensorflow neural network
- Image feature extraction using Inception V3
- Word feature extraction using LSTM RNN
- Image caption model using bidirectional LSTM RNN
- Achieved 60% Correctness (due to training time limitation)
