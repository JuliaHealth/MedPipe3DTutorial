#from https://github.com/maxfreu/SegmentationModels.jl/blob/master/src/unet.jl


using Flux
using Reexport

@reexport using Metalhead

const Classifier = Union{ResNet, VGG}


# trim off unused network parts
get_encoding_network_part(m::Union{ResNet,  VGG}) = m.layers[1][1:end-1]
# get_encoding_network_part(m::MobileNetv1) = m.layers[1][1:end-2]
# get_encoding_network_part(m::MobileNetv2) = m.layers[1][1:end-2][1]
# get_encoding_network_part(m::MobileNetv3) = m.layers[1][1:end-2][1]

function get_encoding_network_part_with_fixed_input_channels(m, input_channels)
    layers = get_encoding_network_part(m).layers

    if input_channels != default_input_size(m)[3]
        new_conv = change_convlayer_input(layers[1], input_channels)
        layers[1] = new_conv
    end

    if prepend_identity(m)
        return Chain(Chain(identity), layers...)
    else
        return Chain(layers...)
    end
end

# if the first conv directly performs downsampling, we link the input directly to to the last decoder stage
# for that we have to prepend the identity layer
prepend_identity(::Classifier) = true
prepend_identity(::VGG) = false

default_input_size(::Classifier) = (256,256,3,1)

function outputsizes(fs::Tuple, x)
    res = Flux.outputsize(first(fs), x)
    return (res, outputsizes(Base.tail(fs), res)...)
end

outputsizes(::Tuple{}, x) = ()
outputsizes(c::Chain, x) = outputsizes(c.layers, x)

function encoder(m, input_channels)
    input_size = default_input_size(m)
    input_size = (input_size[1:2]..., input_channels, input_size[4])
    encoder = get_encoding_network_part_with_fixed_input_channels(m, input_channels)
    os = outputsizes(encoder, input_size)  # snoop intermediate output sizes
    os = map(x->x[1], os)  # take only the width
    os = (os..., 0)  # the below loop checks for differences in size, so we have to set an endpoint to also catch the last layers

    layers = []
    last_size = os[1]
    last = 1
    for i in 1:length(os)
        current_size = os[i]
        if current_size != last_size
            push!(layers, encoder[last:i-1])
            last = i
            last_size = current_size
        end
    end
    return Chain(layers...)
end

function encoder_channels(m, input_channels=3)
    os = outputsizes(m, (512,512,input_channels,1))
    return getindex.(os, 3)
end


function change_convlayer_input(cl::Conv, new_in_channels::Integer)
    w,h,in,out = size(cl.weight)
    new_conv = Conv((w,h), new_in_channels => out, cl.σ; stride=cl.stride, pad=cl.pad, dilation=cl.dilation, groups=cl.groups, bias=cl.bias)
    if new_in_channels > in
        new_conv.weight[:,:,1:in,:] .= cl.weight
    elseif new_in_channels < in
        @views new_conv.weight .= cl.weight[:,:,1:new_in_channels,:]
    end
    return new_conv
end


function double_conv(in_channels, out_channels; activation=relu, bn_momentum=0.1f0)
    Chain(Conv((3,3), in_channels=>out_channels,  activation; pad=1),
          BatchNorm(out_channels; momentum=bn_momentum),
          Conv((3,3), out_channels=>out_channels, activation; pad=1),
          BatchNorm(out_channels; momentum=bn_momentum)
         )
end

struct UNetUpBlock{U,C}
    upsampling_op::U
    conv_op::C
end

Flux.@functor UNetUpBlock

function UNetUpBlock(in_ch_up, in_ch_concat, out_ch; activation=relu, bn_momentum=0.1f0)
    up_op = Upsample(:bilinear; scale=2)
    conv_op = double_conv(in_ch_up + in_ch_concat, out_ch; activation=activation, bn_momentum=bn_momentum)
    UNetUpBlock(up_op, conv_op)
end

function Base.show(io::IO, b::UNetUpBlock)
    println(io, "UpBlock")
    println(io, b.upsampling_op)
    println(io, b.conv_op)
end

function (b::UNetUpBlock)(up_input, concat_input; dims=3)
    up = cat(b.upsampling_op(up_input), concat_input; dims=dims)
    return b.conv_op(up)
end

"""
    UNet(in_channels::Integer=3, num_classes::Integer=1; init_channels::Integer=16, stages::Integer=4, final_activation=sigmoid)
    UNet(m::Classifier; num_classes=1, decoder_channels=(16,32,64,128,256,512,1024), final_activation=sigmoid)

There are two options to instantiate a [UNet](https://arxiv.org/pdf/1505.04597.pdf):

1. with a given number of input channels and output classes
2. with a classifier from Metalhead, either ResNet or VGG
"""
struct UNet{E,D,S}
    encoder::E
    decoder::D
    segmentation_head::S
end

Flux.@functor UNet

# home-made version
"""
    UNet(in_channels::Integer=3, num_classes::Integer=1; init_channels::Integer=16, stages::Integer=4, final_activation=sigmoid)

Instantiates a custom UNet which has double convolutions in each encoder and decoder stage.
"""
function UNet(in_channels::Integer=3, num_classes::Integer=1;
              init_channels::Integer=16,
              stages::Integer=4,
              final_activation=sigmoid)

    down_stage1 = double_conv(in_channels, init_channels)
    down_stages = ntuple(i -> Chain(MaxPool((2,2)),
                                    double_conv(init_channels*2^(i-1), init_channels*2^i)),
                        stages)
    
    encoder = Chain(down_stage1, down_stages...)
    decoder = ntuple(i -> UNetUpBlock(init_channels*2^(stages-i+1),
                                      init_channels*2^(stages-i),
                                      init_channels*2^(stages-i)),
                    stages)

    segmentation_head = Conv((1,1), init_channels=>num_classes, final_activation)
    
    UNet(encoder, decoder, segmentation_head)
end


"""
    UNet(m::Classifier; num_classes=1, decoder_channels=(16,32,64,128,256,512,1024), final_activation=sigmoid, input_channels=3)

Instantiates a UNet based on a given backbone.

## Args
    - m: The backbone, e.g. ResNet()
    - num_classes: Number of output classes
    - final_activation: Final activation layer
    - input_channels: Number of input channels to the network. Changes the first convolution layer of the encoder. 
      The original weight tensor is copied into the newly created one; the remaining weight are initalized in the default way.
"""
function UNet(m::Classifier;
              num_classes=1,
              decoder_channels=(16,32,64,128,256,512,1024),
              final_activation=sigmoid,
              input_channels=3)
    enc = encoder(m, input_channels)
    enc_channels = encoder_channels(enc, input_channels)
    decoder_channels = decoder_channels[1:length(enc_channels)-1]
    decoder_channels = (decoder_channels..., last(enc_channels))
    
    decoder = ntuple(i -> UNetUpBlock(decoder_channels[i+1], enc_channels[i], decoder_channels[i]), length(enc_channels)-1)[end:-1:1]
    
    segmentation_head = Conv((1,1), decoder_channels[1]=>num_classes, final_activation)
    UNet(enc, decoder, segmentation_head)
end

function decode(ops::Tuple, ft::Tuple)
    up = first(ops)(ft[end], ft[end-1])
    decode(Base.tail(ops), (ft[1:end-2]..., up))
end

decode(::Tuple{}, ft::NTuple{1, T}) where T = first(ft)

function (u::UNet)(input)
    encoder_features = Flux.activations(u.encoder, input)
    up = decode(u.decoder, encoder_features)
    return u.segmentation_head(up)
end

function Base.show(io::IO, u::UNet)
    println(io, "UNet:")
    print(io, "\n")
    println(io, "Encoder:")
    Flux._big_show(io, u.encoder)
    
    println(io, "\n")
    println(io, "Decoder:")
    for l in u.decoder
        println(io, l)
    end
    println(io, "Segmentation head:")
    println(io, u.segmentation_head)
end