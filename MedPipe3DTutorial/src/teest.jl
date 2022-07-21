using Pkg
#Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
#Pkg.add(url="https://github.com/maxfreu/SegmentationModels.jl.git")
import BSON,MedPipe3D
using BSON: @save
using BSON: @load
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA
using HDF5,Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using Hyperopt,Plots
using MedPipe3D.LoadFromMonai
#]add ProgressMeter StaticArrays BSON Distributed Flux Hyperopt Plots MedEye3d Distributions Clustering IrrationalConstants ParallelStencil CUDA HDF5 MedEval3D MedPipe3D Colors
CUDA.allowscalar(true)

#directory where we want to store our HDF5 that we will use
pathToHDF5="C:\\projects\\data\\smallDataSet.hdf5"
# data downloaded from http://medicaldecathlon.com/   (task 9 spleen)
#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
#data_dir = "/media/jakub/NewVolume/forJuliaData/spleenData/Task09_Spleen/Task09_Spleen/Task09_Spleen/"
fid = h5open(pathToHDF5, "w")
#representing number that is the patient id in this dataset
patentNum = 0#12
patienGroupName=string(patentNum)
z=8# how big is the area from which we collect data to construct probability distributions
klusterNumb = 5# number of clusters - number of probability distributions we will use
#******************for display
#just needed so we will not have 2 same colors for two diffrent informations
listOfColorUsed= falses(18)
const USE_GPU = true


data_dir = "C:\\projects\\CtOrg"

train_labels = map(fileEntry-> joinpath(data_dir,"labels",fileEntry),readdir(joinpath(data_dir,"labels"); sort=true))
train_images = map(fileEntry-> joinpath(data_dir,"volumes",fileEntry),readdir(joinpath(data_dir,"volumes"); sort=true))
#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
# the paths are for some reasons reading incorrectly
# zipped=map(tupl -> (replace(tupl[1], "._" => ""), replace(tupl[2], "._" => "")),zipped)

tupl=zipped[patentNum]
#proper loading using some utility function
targetSpacing=(1,1,1)

loaded = LoadFromMonai.loadBySitkromImageAndLabelPaths(tupl[1],tupl[2],targetSpacing)


imageArr=Int32.(loaded[1])
# loaded = LoadFromMonai.LoadFromMonai.loadandPad(
#     tupl[1],tupl[2]
#     ,targetSpacing
#     ,(1111,1111,1111))


#!!!!!!!!!! important if you are just creating the hdf5 file  do it with "w" option otherwise do it with "r+"
#fid = h5open(pathToHDF5, "r+") 
gr= getGroupOrCreate(fid, patienGroupName)

labelArr=map(entry-> UInt32(entry==1),loaded[2])
#we save loaded and trnsformed data into HDF5 to avoid doing preprocessing every time
saveMaskBeforeVisualization(fid,patienGroupName,imageArr,"image", "CT" )
saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )

# here we did default transformations so voxel dimension is set to 1,1,1 in any other case one need to set spacing attribute manually to proper value
# spacing can be found in metadata dictionary that is third entry in loadByMonaiFromImageAndLabelPaths output
# here metadata = loaded[3]
writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])







##below we define additional arrays that are not present in original data but will be needed for annotations and storing algorithm output 

#manual Modification array
#manual Modification array
manualModif = MedEye3d.ForDisplayStructs.TextureSpec{UInt32}(# choosing number type manually to reduce memory usage
    name = "manualModif",
    color = RGB(0.2,0.5,0.2) #getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
    ,minAndMaxValue= UInt32.([0,1]) #important to keep the same number type as chosen at the bagining
    ,isEditable = true ) # we will be able to manually modify this array in a viewer

algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
    name = "algoOutput",
    # we point out that we will supply multiple colors
    isContinuusMask=true,
    colorSet = [RGB(1.0,0.0,0.0),RGB(1.0,1.0,0.0) ]
    ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
   )

    addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,2)
    addTextSpecs[1]=manualModif
    addTextSpecs[2]=algoVisualization




    mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)








    #manual Modification array
manualModif = MedEye3d.ForDisplayStructs.TextureSpec{UInt32}(# choosing number type manually to reduce memory usage
name = "manualModif",
color = RGB(0.2,0.5,0.2) #getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
,minAndMaxValue= UInt32.([0,1]) #important to keep the same number type as chosen at the bagining
,isEditable = true ) # we will be able to manually modify this array in a viewer

labelSet = MedEye3d.ForDisplayStructs.TextureSpec{UInt32}(
name = "labelSet",
# we point out that we will supply multiple colors
isContinuusMask=true,
colorSet = [RGB(1.0,0.0,0.0),RGB(1.0,0.0,0.0),RGB(1.0,1.0,0.0)]
,minAndMaxValue= UInt32.([0,2])# values between 0 and 1 as this represent probabilities
)   
addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,3)
addTextSpecs[1]=manualModif
addTextSpecs[2]=algoVisualization
addTextSpecs[3]=labelSet


#2) primary display of chosen image 
mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)
mainScrollDat.dataToScroll




group = fid[patienGroupName]
attrName="spacing"

value=HDF5.read_attribute(group, attrName)
delete_attribute(group,attrName)
write_attribute(group, attrName, value)






delete_attribute(group,"spacing")
writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])
mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)
mainScrollDat.dataToScroll







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
calculatePatchStatistics(allNeededCoord,I
"""
function calculatePatchStatistics(a ::Type{Numb},b ::Type{myFloat},allNeededCoord ,I )  where{Numb, myFloat}
    return [ [getSampleMeanAndStd(a,b, x,I) for x in outer ] for outer in  allNeededCoord]
end









#we load image from displayed object
image= getArrByName("image" ,mainScrollDat)
manualModif= getArrByName("manualModif" ,mainScrollDat)
manualModifPrim=deepcopy(manualModif)
maximum(manualModif) # it should be greater than 0 if you marked anything in array
##coordinates of manually set points
coordsss= getCoordinatesOfMarkings(eltype(image),eltype(manualModif),  manualModif, image) |>
    (seedsCoords) ->getPatchAroundMarks(seedsCoords,z ) |>
    (patchCoords) ->allNeededCoord(patchCoords,z )

#getting patch statistics - mean and covariance
patchStats = calculatePatchStatistics(eltype(image),Float64, coordsss, image)

#separate distribution for each marked point
distribs = map(patchStat-> fit(MvNormal, reduce(hcat,(patchStat)))  , patchStats  )








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

mainArrSize= size(image)
# for simplicity not using the occupancy API - in production one rather should
threads=(8,4,8)
blocks = (cld(mainArrSize[1],threads[1]), cld(mainArrSize[2],threads[2])  , cld(mainArrSize[3],threads[3]))



algoOutput= getArrByName("algoOutput" ,mainScrollDat)
mainArrSize= size(image)

algoOutputGPU=CuArray(algoOutput)
imageGPU=CuArray(image)
conGPU = CuArray(allConstants)
@cuda threads=threads blocks=blocks applyGaussKernel(conGPU,imageGPU,mainArrSize,algoOutputGPU, 5,klusterNumb)
copyto!(algoOutput,algoOutputGPU)
sum(algoOutput)# just to check is anything copied

# copyto!(algoOutput,algoOutputGPU)
algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
maxEl = maximum(algoOutputGPU)
algoOutputB[:,:,:]=algoOutput./maxEl
algoOutputGPU=algoOutputGPU./maxEl
visualizationFromHdf5.refresh(MedEye3d.SegmentationDisplay.mainActor) 



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

#rate=mean(algoOutputB)
rate=0.03
relaxLabels(algoOutputGPU,4,rate)
copyto!(algoOutput,algoOutputGPU)
sum(algoOutput)# just to check is anythink copied



algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
algoOutputB[:,:,:]=algoOutput
visualizationFromHdf5.refresh(MedEye3d.SegmentationDisplay.mainActor) 


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


algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
maxEl = maximum(algoOutputGPU)
algoOutputB[:,:,:]=algoOutput
visualizationFromHdf5.refresh(MedEye3d.SegmentationDisplay.mainActor) 



conf= ConfigurtationStruct(md=true, dice=true)
numberToLookFor = 1.0
liverGold= getArrByName("labelSet" ,mainScrollDat)
preparedDict=MedEval3D.MainAbstractions.prepareMetrics(conf)
calculateAndDisplay(preparedDict,mainScrollDat, conf, numberToLookFor,CuArray(liverGold),algoOutputGPU )


# algoOutputB= getArrByName("algoOutput" ,mainScrollDat)
# maxEl = maximum(algoOutputGPU)
# algoOutputB[:,:,:]=algoOutput./maxEl
visualizationFromHdf5.refresh(MedEye3d.SegmentationDisplay.mainActor) 


saveManualModif(fid,patienGroupName , mainScrollDat)

close(fid)



conf= ConfigurtationStruct(md=true, dice=true)
numberToLookFor = 1.0
liverGold= getArrByName("labelSet" ,mainScrollDat)
CUDAGold=CuArray(liverGold)
preparedDict=MedEval3D.MainAbstractions.prepareMetrics(conf)
res= calcMetricGlobal(preparedDict,conf,CUDAGold,algoOutputGPU,numberToLookFor)
res