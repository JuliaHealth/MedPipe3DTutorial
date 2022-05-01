# MedPipe3DTutorial
Showing how to use MedPipe (including MedEye3d  MedEval3D  HDF5 and MONAI preprocessing)


# First example
Example is based on public dataset that can be found in link below

https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations

labels without readme file should be put in the labels folder and volumes in "volumes 0-49" folder - there should be the same number of labels as volumes - you do not need to download all of them if you do not want to. 


Next we establish imports
In case any of this packages are not already installed do it now

```
using Pkg
#Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")

using MedPipe3D
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA
using HDF5
```

We define some constants

```

#representing number that is the patient id in this dataset
patienGroupName="0"
z=7# how big is the area from which we collect data to construct probability distributions
klusterNumb = 5# number of clusters - number of probability distributions we will use


#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
data_dir = "D:\\dataSets\\CTORGmini\\"
#directory where we want to store our HDF5 that we will use
pathToHDF5="D:\\dataSets\\forMainHDF5\\smallLiverDataSet.hdf5"
```
Finding paths to files and constructing tuples with them - check weather they will point to the same patient

```
train_labels = map(fileEntry-> joinpath(data_dir,"labels",fileEntry),readdir(joinpath(data_dir,"labels"); sort=true))
train_images = map(fileEntry-> joinpath(data_dir,"volumes 0-49",fileEntry),readdir(joinpath(data_dir,"volumes 0-49"); sort=true))


#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
tupl=zipped[1]
tupl
```
for example in my case tuple of directories pointing to the patient 0 looks like below
("D:\\dataSets\\CTORGmini\\volumes 0-49\\volume-0.nii.gz", "D:\\dataSets\\CTORGmini\\labels\\labels-0.nii.gz")

Next we load and preprocess data - here I use defoult transformations - If You want to use some other transformations you can supply third argument to loadByMonaiFromImageAndLabelPaths function more in LoadFromMonai.loadByMonaiFromImageAndLabelPaths function doc
```
loaded = LoadFromMonai.loadByMonaiFromImageAndLabelPaths(tupl[1],tupl[2])
```

Next we open our persistent HDF5 file system and save transformed and preprocessed data there so we will not need to preprocess it every time we invoke algorithm. Also what is important is if we are just creating hdf5 file we use 
h5open(pathToHDF5, "w") and if it already exist fid = h5open(pathToHDF5, "r+") 


```
#!!!!!!!!!! important if you are just creating the hdf5 file  do it with "w" option otherwise do it with "r+"
fid = h5open(pathToHDF5, "w")
#fid = h5open(pathToHDF5, "r+") 
gr= getGroupOrCreate(fid, patienGroupName)

#for this particular example we are intrested only in liver so we will keep only this label
labelArr=map(entry-> UInt32(entry==1),loaded[2])


#we save loaded and trnsformed data into HDF5 to avoid doing preprocessing every time
saveMaskBeforeVisualization(fid,patienGroupName,loaded[1],"image", "CT" )
saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )

# here we did default transformations so voxel dimension is set to 1,1,1 in any other case one need to set spacing attribute manually to proper value
# spacing can be found in metadata dictionary that is third entry in loadByMonaiFromImageAndLabelPaths output
# here metadata = loaded[3]
writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])

```
Now we prepare definitions that will be used to generate additional arrays that we will use as output of our algorithm and imput to manual modifications.

```
#just needed so we will not have 2 same colors for two diffrent informations
listOfColorUsed= falses(18)

##below we define additional arrays that are not present in original data but will be needed for annotations and storing algorithm output 

#manual Modification array
manualModif = MedEye3d.ForDisplayStructs.TextureSpec{UInt8}(# choosing number type manually to reduce memory usage
    name = "manualModif",
    color =getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
    ,minAndMaxValue= UInt8.([0,1]) #important to keep the same number type as chosen at the bagining
    ,isEditable = true ) # we will be able to manually modify this array in a viewer

algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
    name = "algoOutput",
    # we point out that we will supply multiple colors
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
    ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
   )

    addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,2)
    addTextSpecs[1]=manualModif
    addTextSpecs[2]=algoVisualization

```

Time to display data (if you do not see anything make sure that viewer is active window and scroll up and down)

```
mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)

```
after some scrolling (you can press f for fast scrolling) - more details how to operate viewer in https://github.com/jakubMitura14/MedEye3d.jl

You should see sth like below

![image](https://user-images.githubusercontent.com/53857487/160683323-efc0efae-3742-4ec1-b330-dbdde6dc1075.png)


Now time for anotations - just drag the mouse pressing left mouse key over the liver you will get sth like below

![image](https://user-images.githubusercontent.com/53857487/160683751-27983862-c04b-4dee-bc3f-b0bf397b4c74.png)


# Example algorithm
Time for proper algorithm - we will use data from you annotations to construct ptobability distributions that will summarize local liver properties.

```

"""
works only for 3d cartesian coordinates
  cart - cartesian coordinates of point where we will add the dimensions ...
"""
function cartesianTolinear(pointCart::CartesianIndex{3}) :: Int16
   abs(pointCart[1])+ abs(pointCart[2])+abs(pointCart[3])
end


"""
point - cartesian coordinates of point around which we want the cartesian coordeinates
return set of cartetian coordinates of given distance -patchSize from a point
"""
function cartesianCoordAroundPoint(pointCart::CartesianIndex{3}, patchSize ::Int)
ones = CartesianIndex(patchSize,patchSize,patchSize) # cartesian 3 dimensional index used for calculations to get range of the cartesian indicis to analyze
out = Array{CartesianIndex{3}}(UndefInitializer(), 6+2*patchSize^4)
index =0
for J in (pointCart-ones):(pointCart+ones)
  diff = J - pointCart # diffrence between dimensions relative to point of origin
    if cartesianTolinear(diff) <= patchSize
      index+=1
      out[index] = J
    end
    end
return out[1:index]
end


"""
By iteratively  searching through the mask M array cartesian coordinates of all entries with value 7 will be returned.
Important the number 7 is completely arbitrary - and need to agree with the number set in the annotator
"""
function getCoordinatesOfMarkings(::Type{ImageNumb}, ::Type{maskNumb}, M, I )  ::Vector{CartesianIndex{3}} where{ImageNumb,maskNumb}
    return filter((index)->M[index]>0 ,CartesianIndices(M))
end

"""
We need to define the patch Ω using getCartesianAroundPoint around each seed point - we will list of coordinates set  
markings - calculated  earlier in getCoordinatesOfMarkings  z is the size of the patch - it is one of the hyperparameters
return the patch of pixels around each marked point
"""
function getPatchAroundMarks(markings ::Vector{CartesianIndex{3}}, z::Int) 
    return [cartesianCoordAroundPoint(x,z) for x in markings]
end    

"""
6.Now we apply analogical operation to each point coordinates of each patch  to get set of sets of sets where the nested sub patch will be referred to as Ω_ij
markingsPatches is just the output of getPatchAroundMarks 
z is the size of the patch - it is one of the hyperparameters
return nested patches so we have patch around each voxel from primary patch
"""
function allNeededCoord(markingsPatches ,z::Int ) ::Vector{Vector{Vector{CartesianIndex{3}}}}
    return [getPatchAroundMarks(x,z) for x in markingsPatches]
end  



"""
We define function that give set of cartesian coordinates  returns the vector where first entry is a sample mean and second one sample standard deviation 
 of values in image I in given coordinates
 first type is specyfing the type of number in image array second in the output - so we can controll what type of float it would be
getSampleMeanAndStd(points,I)
"""
function  getSampleMeanAndStd(a ::Type{Numb},b ::Type{myFloat}, coords::Vector{CartesianIndex{3}} , I  ) ::Vector{myFloat} where{Numb, myFloat}

  sizz = size(I)  
  arr= I[filter(c-> c[1]>0 && c[2]>0 && c[3]>0 
                && c[1]<sizz[1]&& c[2]<sizz[2] && c[3]<sizz[3]  ,coords)]
                
    return [mean(arr), std(arr)]   
end

"""
Next we reduce each of the sub patch omega using getSampleMeanAndStd function and store result in patchStats
calculatePatchStatistics(allNeededCoord,I\
"""
function calculatePatchStatistics(a ::Type{Numb},b ::Type{myFloat},allNeededCoord ,I )  where{Numb, myFloat}
    return [ [getSampleMeanAndStd(a,b, x,I) for x in outer ] for outer in  allNeededCoord]
end

#we load image from displayed object
image= getArrByName("image" ,mainScrollDat)
manualModif= getArrByName("manualModif" ,mainScrollDat)

maximum(manualModif) # it should be greater than 0 if you marked anything in array
##coordinates of manually set points
coordsss= getCoordinatesOfMarkings(eltype(image),eltype(manualModif),  manualModif, image) |>
    (seedsCoords) ->getPatchAroundMarks(seedsCoords,z ) |>
    (patchCoords) ->allNeededCoord(patchCoords,z )

#getting patch statistics - mean and covariance
patchStats = calculatePatchStatistics(eltype(image),Float64, coordsss, image)

#separate distribution for each marked point
distribs = map(patchStat-> fit(MvNormal, reduce(hcat,(patchStat)))  , patchStats  )


```
In order to reduce computational complexity  we will reduce the number of used distributions using kl divergence



```

#we are comparing all distributions 
klDivs =map(outerDist->    map(dist->kldivergence( outerDist  ,dist), distribs  ), distribs  )
klDivsInMatrix = reduce(hcat,(klDivs))
#clustering with kmeans
R = kmeans(klDivsInMatrix, klusterNumb; maxiter=200, display=:iter)

#now identify indexes for some example distributions from each cluster
indicies = zeros(Int64,klusterNumb )
a = assignments(R) # get the assignments of points to clusters
for i in 1:klusterNumb
    for j in 1:length(distribs)
        if(a[j] == i)
            indicies[i]=j
        end
    end    
end
indicies

#ditributions from diffrent clusters
chosenDistribs = map(ind->distribs[ind] ,indicies)



```
Preparing set of constants needed for application of the calculated set of probability distribution to the image

```

### getting constants from distributions

"""
calculate log normalization constant from distribution
"""
function mvnormal_c0(d::AbstractMvNormal)
    ldcd = logdetcov(d)
    return - (length(d) * oftype(ldcd, log2π) + ldcd) / 2
end

"""
get constants needed for applying probability distributions
 return vector   1) logConst 2) mu1 3) mu2 4) invcov00 5)invcov01 6)invcov10 7)invcov11 
"""
function getDistrConstants(exampleDistr)
    c0= mvnormal_c0(exampleDistr)
    invCov= inv(exampleDistr.Σ)
    return [c0,exampleDistr.μ[1],exampleDistr.μ[2],invCov[1,1],invCov[1,2],invCov[2,1],invCov[2,2]  ]
end#getDistrConstants


# creating matrix from constants
allConstants = map(distr-> getDistrConstants(distr)  , chosenDistribs) |>
               (vectOfvects)-> reduce(hcat, vectOfvects)

```
## GPU algorithms

Till this moment our algorithm was for simplicity run sequentially on CPU. Hovewer now the evaluation of whole set of probability distributions for each voxel requires parallelization. The most effective parallelization can be achieved using GPU architecture for which Julia has excellent support via couple libraries what will be presented below.

Evaluation of chosen probability distributions at each given point using CUDA.jl
```
mainArrSize= size(image)

"""
utility macro to iterate in given range around given voxel
"""
macro iterAround(ex   )
    return esc(quote
        for xAdd in -r:r
            x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))+xAdd
            if(x>0 && x<=mainArrSize[1])
                for yAdd in -r:r
                    y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))+yAdd
                    if(y>0 && y<=mainArrSize[2])
                        for zAdd in -r:r
                            z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))+zAdd
                            if(z>0 && z<=mainArrSize[3])
                                if((abs(xAdd)+abs(yAdd)+abs(zAdd)) <=r)
                                    $ex
                                end 
                            end
                        end
                    end    
                end    
            end
        end    
    end)
end
      

"""
con - matrix of precalculated constants
image - main image here computer tomography image
mainArrSize - dimensions of image
output - where we want to save the calculations
r - size of the evaluated patch
klusterNumb- number of clusters - number of probability distributions we will use
"""
function applyGaussKernel(con,image,mainArrSize,output, r::Int,klusterNumb::Int)
    for probDist in 1:klusterNumb
        summ=0.0
        sumCentered=0.0
        lenn= UInt8(0)
        #get mean
        @iterAround begin 
            lenn=lenn+1
            summ+=image[x,y,z]    
        end
        summ=summ/lenn
        #get standard deviation
        @iterAround sumCentered+= ((image[x,y,z]-summ )^2)

        #here we have standard deviation
        sumCentered= sqrt(sumCentered/(lenn-1))
        #centering - subtracting means...
        summ=summ-con[2,probDist]
        sumCentered=sumCentered-con[3,probDist]
        #saving output
        x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
        y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))
        z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))
        if(x>0 && x<=mainArrSize[1] && y>0 && y<=mainArrSize[2] &&z>0 && z<=mainArrSize[3] )
            output[x,y,z]=  max(exp(con[1,probDist]-( ((summ*con[4,probDist]+sumCentered*con[6,probDist])*summ+(summ*con[5,probDist]+sumCentered*con[7,probDist])*sumCentered)/2 ) ),output[x,y,z]  )
        end  
    end#for
    return
end#main kernel

# for simplicity not using the occupancy API - in production one rather should
threads=(8,4,8)
blocks = (cld(mainArrSize[1],threads[1]), cld(mainArrSize[2],threads[2])  , cld(mainArrSize[3],threads[3]))
using CUDA
algoOutputGPU=CuArray(algoOutput)
imageGPU=CuArray(image)
conGPU = CuArray(allConstants)
@cuda threads=threads blocks=blocks applyGaussKernel(conGPU,imageGPU,mainArrSize,algoOutputGPU, 5,klusterNumb)
#@cuda threads=threads blocks=blocks applyGaussKernel(conGPU,imageGPU,mainArrSize,algoOutputGPU, z,klusterNumb)
copyto!(algoOutput,algoOutputGPU)
sum(algoOutput)# just to check is anythink copied

#copy and divide by max so will be easier to visualize
algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
maxEl = maximum(algoOutputGPU)
algoOutputB[:,:,:]=algoOutput./maxEl


```

Now after you scroll up and down you should see sth like below

![image](https://user-images.githubusercontent.com/53857487/160751008-763e6b9e-edd0-4196-8b5f-fd6ed466cc75.png)

in green and yellow you can see output of our function in order to see it better - we can press ctrl+1 to make the gold standard invisible
Number depends on order by which given mask was supplied - you can make sure the you are setting visibility to correct mask by pressing ctrl+ number and looking at the log output in your repl - for example We named gold standard mask as labelSet and the information in repl after pressing ctrl + number for label set 
```
julia> [ Info:  set visibility of labelSet to false
```
![image](https://user-images.githubusercontent.com/53857487/160758235-30b9119a-a806-441e-81cc-22cf2d765628.png)




To make the labelSet visible just press shift+1
Remember that for any shortcut to work you need to have Segmentation visualization viewer as an active window.


Now we can choose to save the output of our algorithm to the HDF5 iw we wan to by 

```
saveMaskbyName(fid,patienGroupName , mainScrollDat, "algoOutput","boolLabel")
```

Next stage of proposed algorthm is relaxation labelling that also will be done with GPU acceleration but now using ParallelStencil

```

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

@init_parallel_stencil(CUDA, Float64, 3);


#cutoff set manually to rate
@parallel_indices (ix,iy,iz) function relaxationLabellKern(In, rate)
    # 7-point Neuman stencil
    if (ix>1 && iy>1 && iz>1 &&      ix<(size(In,1))&& iy<(size(In,2)) && iz<(size(In,3)))
        In[ix,iy,iz] = ( (In[ix-1,iy  ,iz  ] >rate)+
                          (In[ix-1,iy  ,iz  ]>rate)+ (In[ix+1,iy  ,iz  ]>rate) +
                          (In[ix  ,iy-1,iz  ]>rate) + (In[ix  ,iy+1,iz  ]>rate) +
                          (In[ix  ,iy  ,iz-1]>rate) + (In[ix  ,iy  ,iz+1]>rate) )/7.0

     
    end
    return
end

@views function relaxLabels(In, iterNumb,rate)
    # Calculation
    for i in 1:iterNumb
        innerRate=rate + ((i/iterNumb)/3)
        @parallel relaxationLabellKern(In,innerRate)
    end#for    
    return
end

rate=0.05
relaxLabels(algoOutputGPU,19,rate)


Int(round(sum(algoOutputGPU)))# just to check is data there



copyto!(algoOutput,algoOutputGPU)
Int(round(sum(algoOutput)))# just to check is anythink copied  #85162

#copy and divide by max so will be easier to visualize
algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
algoOutputB[:,:,:]=algoOutput


```

After scroll you should see sth like below. Outout of the algortithm is yellow, blue is gold standard - How it should look like ideally.



![image](https://user-images.githubusercontent.com/53857487/160756310-cce6847a-1672-43e6-8a88-cfb749e60183.png)



Now we see that our algorithm is approximately showing us where the liver is but is far from perfect. Hovewer to establish how far we need to measure it. Choosing correct segmentation metric can be challenging and I strongly recommend the excellent work bt Taha[1] for which the link is below for intrested users. In this case for presentation we will calculate 
Mahalanobis distance and Dice similarity coefficient.
!!! important MedEval3D requires CUDA.jl to work.

```

# first we need to define the cutoff  over which we will decide that probability indicates that it is truly a liver 
#####simple  tresholding
function tresholdingKernel(mainArrSize,output)
  
        x= (threadIdx().x+ ((blockIdx().x -1)*CUDA.blockDim_x()))
        y= (threadIdx().y+ ((blockIdx().y -1)*CUDA.blockDim_y()))
        z= (threadIdx().z+ ((blockIdx().z -1)*CUDA.blockDim_z()))
        if(x>0 && x<=mainArrSize[1] && y>0 && y<=mainArrSize[2] &&z>0 && z<=mainArrSize[3] )
            output[x,y,z]=  (output[x,y,z]>0.5)
        end  
    return
end#main kernel

# for simplicity not using the occupancy API - in production one rather should
threads=(8,8,8)
blocks = (cld(mainArrSize[1],threads[1]), cld(mainArrSize[2],threads[2])  , cld(mainArrSize[3],threads[3]))
@cuda threads=threads blocks=blocks tresholdingKernel(mainArrSize,algoOutputGPU)



using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
conf= ConfigurtationStruct(md=true, dice=true)
numberToLookFor = 1.0
liverGold= getArrByName("liver" ,mainScrollDat)

preparedDict=MedEval3D.MainAbstractions.prepareMetrics(conf)
calculateAndDisplay(preparedDict,mainScrollDat, conf, numberToLookFor,CuArray(liverGold),algoOutputGPU )




mainScrollDat.mainTextToDisp

copyto!(algoOutput,algoOutputGPU)
Int(round(sum(algoOutput)))# just to check is anythink copied  #85162
#copy and divide by max so will be easier to visualize
algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
algoOutputB[:,:,:]=algoOutput

```

Scroll up and down and in the top left corner you should see displayed requested metrics

![image](https://user-images.githubusercontent.com/53857487/160757193-14f46edf-de3d-47be-ba1a-0795f7b7a327.png)

Now is time toreiterate , try to tweak hyperparameters, use other algorithm etc. ... What is most important as presented above one is free to use basically any Julia package, as you probably now signifacant amount of algorithms are implemented only in Julia and not in other programming languages, hence they were not applied to medical image segmentation task yet, so the only thing needed is an idea, becouse tools are already here :). 




Lastly we can save our output and close the connection to HDF5

```
saveManualModif(fid,patienGroupName , mainScrollDat)

close(fid)
```

If You will find usefull my work please cite it 

```
@Article{Mitura2021,
  author   = {Mitura, Jakub and Chrapko, Beata E.},
  journal  = {Zeszyty Naukowe WWSI},
  title    = {{3D Medical Segmentation Visualization in Julia with MedEye3d}},
  year     = {2021},
  number   = {25},
  pages    = {57--67},
  volume   = {15},
  doi      = {10.26348/znwwsi.25.57},
  keywords = {OpenGl, Computer Tomagraphy, PET/CT, medical image annotation, medical image visualization},
}

```


[1] https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-015-0068-x



