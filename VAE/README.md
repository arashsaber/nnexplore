## Variational Autoencoder

A convolutional autoencoder object. 
To use it, simply call the object as below.

**Getting the data:**
We start by downloading the MNIST dataset:

    import tflearn.datasets.mnist as mnist
    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)
    trainX = trainX.reshape([-1, 28, 28])
    testX = testX.reshape([-1, 28, 28])


**Build the model, train/save or load:**

    vae = VAE()

    # train and save the model
    vae.train(trainX, testX, n_epoch=10)
    vae.save('./VAE/saved_models/model.tfl')

    # load the model
    vae.load('./VAE/saved_models/model.tfl')


**Testing the generator:**
  
    plt.imshow(vae.generator_viewer(128), cmap='gray')

<img src="https://github.com/arashsaber/nnexplore/blob/lenovo/VAE/Figs/generated.png" width="600">

**Testing the dimensionality reduction:**
  
    z = vae.reduce_dimension(trainX[10:15,:,:])

**Testing the reconstruction:**

    plt.imshow(np.hstack((trainX[10,:,:].reshape(28,28), 
                        vae.reconstruct(trainX[10,:,:]).reshape(28,28)
                        )), cmap='gray')

<img src="https://github.com/arashsaber/nnexplore/blob/lenovo/VAE/Figs/reconstructed0.png" width="400">

    plt.imshow(vae.reconstructor_viewer(trainX[:128,:,:]), cmap='gray')

<img src="https://github.com/arashsaber/nnexplore/blob/lenovo/VAE/Figs/reconstructed.png" width="600">   
    
**Testing the 2D-visualizations:**
Let us now test the 2D viisualization through VAEs:
    
    # build the model
    vae2d = VAE(reduced_dim=2)

    # training and saving
    vae2d.train(trainX, testX, n_epoch=10)
    vae2d.save('./VAE/saved_models/model2d.tfl')
    
    # load the model
    vae2d.load('./VAE/saved_models/model2d.tfl')
    
    
**Displaying the scatter plot of 2d latent features:**

    vae2d.visualization_2d(testX[:1000,:,:], testY[:1000,:])

<img src="https://github.com/arashsaber/nnexplore/blob/lenovo/VAE/Figs/scatterplot.png" width="600">

**Displaying the spectrum of the generated images:**

    vae2d.spectum_2d(25)

<img src="https://github.com/arashsaber/nnexplore/blob/lenovo/VAE/Figs/spectrum.png" width="600">