import numpy as np
from matplotlib import pyplot as plt
#import math
#import random
import SMO_functions as SMO



###The initialization of the parameter in the optimization%%%%%%
N=80;   #Mask dimension in pixels
N_filter=21;   #Amplitude impulse response dimension
pixel=15*1;   #Pixel size (nm)
NA=1.25*1;   #Numerical aperture
lamda=193;   #Wavelength (nm)
order=1;   #Order of Bessel function (assuming step function truncation of diffraction in Fourier plane)
sigma_large_inner=0.5;   #Inner Partial coherence factor
sigma_large_outer=0.6;   #Outer Partial coherence factor
N_coherence=18+0;   #Source dimension


 
target=np.zeros((N,N)) #Initial Target and initial mask
target[:,16:37]=1
target[:,48-5:64]=1
target[0:15,:]=0
target[65:80,:]=0
target[10:70,20:25]=1
target[10:72,20:22]=1

target[55:60,10:22]=1

target[20:50,1:25]=0


source_0=SMO.initial_source(sigma_large_inner,sigma_large_outer,NA,N_coherence,pixel,N,lamda) 
h,g=SMO.amplitude_response_func(N_filter,NA,lamda,pixel,order)
aerial=SMO.calculate_aerial_image(source_0,target,N_filter,NA,lamda,pixel,order,h,g)


################### Initialize optimization
######### Source 
source=SMO.initial_source(sigma_large_inner,sigma_large_outer,NA,N_coherence,pixel,N,lamda) 

#####mask
m=target

flag_mask=np.zeros((N,N));   #Locations of the changable pixels on mask in the current iteration
#flag_source=np.zeros((N_coherence,N_coherence));   #Locations of the changable pixels on source in the current iteration
error=100000;   #Output pattern error in the current iteration

###########Calculate the output pattern error in the previous iteration###########
aerial_pre=SMO.calculate_aerial_image(source,m,N_filter,NA,lamda,pixel,order,h,g)
error_pre=SMO.compute_error(source,m,N_filter,NA,lamda,pixel,order,h,g,target) #Output pattern error in the previous iteration

m=1*target

error_post=0
repeats=0

all_error=[]

plt.imshow(m)

for num_iterations in range(60):
    num_pixels=3000      #number of pixels to change per iteration
        
    m_iter=1*m  #Initialize optimization to output of previous loop
    
    plt.imshow(m_iter, interpolation='none')
    plt.title('Best mask')
    plt.colorbar()
    plt.clim(-2,2)
    plt.show()  

    #flag_mask=0*m_iter+1        #Consider all pixels on mask as candidates
    direction_mask=1*SMO.mask_gradient(source,m_iter,target,aerial,h,g,N_filter,pixel)  #Calculate new mask gradient
    #test_mask=abs(np.multiply(direction_mask,flag_mask))    #abs
    test_mask=abs(direction_mask)
    
    num2search=num_pixels
    a=test_mask
    
    idx = np.argsort(test_mask.ravel())[-1*num2search:][::-1] #Sort pixels in order of gradient value
    topN_val = test_mask.ravel()[idx]
    row_col = np.c_[np.unravel_index(idx, test_mask.shape)]
    row_col=row_col[::-1]

        
    for k in range (row_col.shape[0]):           # Take the first num_pixels pixels with largest gradient value. Flip 1 by 1. Check if cost improves 
        if repeats>2:                           
            row_col=row_col[1:]
        if k>=row_col.shape[0]-1:
            break
            
        i=row_col[k][0]     #set pixel of interest to pixel with largest gradient value
        j=row_col[k][1]
        

        m_iter[i,j]=m_iter[i,j]+0.2*direction_mask[i,j];        # Flip pixel sign at position of max gradient
        flag_mask[i,j]=0        # Don't consider this pixel in subsequent optimizations
                        
        aerial=SMO.calculate_aerial_image(source,m_iter,N_filter,NA,lamda,pixel,order,h,g)   #calculate new AI
        error_post=SMO.compute_error(source,m_iter,N_filter,NA,lamda,pixel,order,h,g,target) #Output pattern error
            
        # complexity_penalty=0.005*np.sum(abs(m_iter))  #L1 regularization
        complexity_penalty=0
        #print(complexity_penalty)
        
        error_post=error_post+complexity_penalty  #add additional cost for new complexity
            
        if (error_post>=error_pre):
             repeats+=1;
        if error_post<error_pre:
            m=1*m_iter
            error_pre=1*error_post
            repeats=0
            
        print('Iteration = '+str(num_iterations)+'     Error = '+ str(error_pre))  
        all_error.append(error_pre)
    
 

    
plt.imshow(target, interpolation='none')
plt.title('Target')
plt.colorbar()
plt.clim(0,2) 
plt.show()

plt.imshow(aerial_pre, interpolation='none')
plt.title('Inital AI')
plt.colorbar()
plt.clim(0,2) 
plt.show()


plt.imshow(np.real(m_iter), interpolation='none')
plt.title('Final mask')
plt.colorbar()
plt.clim(-2,2) 
plt.show()

plt.imshow(np.real(aerial), interpolation='none')
plt.title('Final AI image')
plt.colorbar()
plt.clim(0,2) 
plt.show()


plt.imshow(abs(target-np.real(aerial_pre)), interpolation='none')
plt.title('Target - Initial AI')
plt.colorbar()
plt.clim(0,1) 
plt.show()


plt.imshow(abs(target-np.real(aerial)), interpolation='none')
plt.title('Target - final AI')
plt.colorbar()
plt.clim(0,1) 
plt.show()


# plt.imshow(1*(flag_mask), interpolation='none')
# plt.colorbar()
# plt.title('Mask Edges')
# plt.show()


plt.imshow(1*(direction_mask), interpolation='none')
plt.title('MaskGradient')
plt.colorbar()
plt.show()


plt.plot(np.real(aerial[40,:]),'r' );
plt.plot(np.real(aerial_pre[40,:]),'k' );
plt.plot(np.real(target[40,:]),'b' );
plt.show()


plt.plot(np.log10(all_error),'b' );
plt.show()


