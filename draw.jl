using Knet
using ArgParse
using AutoGrad
using GZip
using Images
using ImageMagick

data_dir = ""
read_attn = true
write_attn = true

A=28 #image width
B=28 #image height
img_size = B*A #canvas size
enc_size = 256 #number of hidden units / output size in LSTM
dec_size = 256
read_n = 5 #read glimpse grid width/height
write_n = 5 #write glimpse grid width/height
read_size = 0
write_size = 0
if read_attn
  read_size = 2*read_n*read_n
else
  read_size = 2*img_size
end

if write_attn
  write_size = write_n*write_n
else
  write_size = img_size
end

z_size = 10 #QSampler output size
T = 10 # MNIST generation sequence length
batch_size = 100 # training minibatch size
train_iters = 10000
learning_rate = 1e-3 # learning rate for optimizer
eps = 1e-8 # epsilon for numerical stability

cs = zeros(1, T)
mus = zeros(1, T)
logsigmas = zeros(1, T)
sigmas = zeros(1, T)

do_share = false
h_dec_prev = zeros(batch_size, dec_size)

enc_state = 0 ##########todo
dec_state = 0 ##########todo

x = zeros(batch_size, img_size)
e = randn(batch_size, z_size)
lstm_enc = 0 ##########todo
lstm_dec = 0 ##########todo

function main(args)
  s = ArgParseSettings()
  @add_arg_table s begin
      ("--outdir"; default=nothing)
  end

  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

  (xtrn,xtst,ytrn,ytst)=loaddata()
  trn = minibatch(xtrn, ytrn, batch_size)
  tst = minibatch(xtst, ytst, batch_size)
  w = initweights()

  if o[:outdir] != nothing
      o[:outdir] = abspath(o[:outdir])
      !isdir(o[:outdir]) && mkdir(o[:outdir])
  end

  optd = map(x->Adam(;lr=learning_rate), w)

  loss1 = test(w,trn)
  @printf("\nepoch: %d, loss: %g\n", 0, loss1)
  if o[:outdir] != nothing
      out = generate(w)
      png = makegrid(out; gridsize=[8,8], scale=2)
      filename = @sprintf("%04d.png",0)
      save(joinpath(o[:outdir],filename), png)
  end

  for epoch = 1:10
      shuffle!(trn)
      loss1 = test(w,trn)
      @printf("epoch: %d, loss: %g\n", epoch, loss1)
      flush(STDOUT)
      if o[:outdir] != nothing
          out = generate(w)
          png = makegrid(out; gridsize=[8,8], scale=2)
          filename = @sprintf("%04d.png",epoch)
          save(joinpath(o[:outdir],filename), png)
      end
  end

  info("Generated files are at the directory: ")
  info(o[:outdir])
end

function linear(w, x, b) ##########todo
  return w*x.+b
end

function minibatch(x, y, batchsize;
    atype=Array{Float32}, xrows=784, yrows=10, xscale=255/2, xnorm=1)
    xbatch(a)=reshape(a./xscale-xnorm, xrows, div(length(a),xrows))
    ybatch(a)=(a[a.==0]=10; sparse(convert(Vector{Int},a),1:length(a),one(eltype(a)),yrows,length(a)))
    xcols = div(length(x),xrows)
    xcols == length(y) || throw(DimensionMismatch())
    data = Any[]
    for i=1:batchsize:xcols-batchsize+1
        j=i+batchsize-1
        push!(data, (xbatch(x[1+(i-1)*xrows:j*xrows]), ybatch(y[i:j])))
    end
    return data
end

function initweights()
    w = Array(Any, 1)
    w[1] = randn(128, 784)
    return w
end

function binary_crossentropy(t,o)
    return -(t*log(o+eps) + (1.0-t)*log(1.0-o+eps))
end

function total_loss()
  x_recons=sigm(cs[end])
  # after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
  Lx=sum(binary_crossentropy(x,x_recons),2) # reconstruction term
  Lx=mean(Lx)

  kl_terms=zeros(1,T)
  Lz=0
  for t in 1:T
    mu2=mus[t]*mus[t]
    sigma2=sigmas[t]*sigmas[t]
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*(mu2+sigma2-2*logsigma)-T*0.5 # each kl term is (1xminibatch)
    KL=sum(kl_terms,1) # this is 1xminibatch, corresponding to summing kl_terms from 1:T ****(add_n in python = sum(x,1) in julia)
    Lz=mean(KL) # average over minibatches                                               ****(reduce_sum(x,1) in python = sum(x,2) in julia)
  end
  cost=Lx+Lz
end

lossgradient = grad(total_loss)

function test(w,data)
  loss1 = 0
    for (x,y) in data
      loss1 = total_loss()
    end
    return loss1/length(data)
end

function generate(w)
    z = tanh(w[1][1:100,1:64]*randn(64,64)) + randn(100,64)
    return min(1,max(0,z))
end

function loaddata()
    info("Loading MNIST...")
    gzload("train-images-idx3-ubyte.gz")[17:end],
    gzload("t10k-images-idx3-ubyte.gz")[17:end],
    gzload("train-labels-idx1-ubyte.gz")[9:end],
    gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = read(f)
    close(f)
    return(a)
end

function makegrid(y; gridsize=[4,4], scale=2.0, shape=(10,10))
    y = reshape(y, shape..., size(y,2))
    y = map(x->y[:,:,x]', [1:size(y,3)...])
    shp = map(x->Int(round(x*scale)), shape)
    y = map(x->Images.imresize(x,shp), y)
    gridx, gridy = gridsize
    outdims = (gridx*shp[1]+gridx+1,gridy*shp[2]+gridy+1)
    out = zeros(outdims...)
    for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:] = 1.0; end
    for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1] = 1.0; end
    x0 = y0 = 2
    for k = 1:length(y)
        x1 = x0+shp[1]-1
        y1 = y0+shp[2]-1
        out[x0:x1,y0:y1] = y[k]

        y0 = y1+2
        if k % gridy == 0
            x0 = x1+2
            y0 = 2
        else
            y0 = y1+2
        end
    end

    return convert(Array{Float64,2}, map(x->isnan(x)?0:x, out))
end

function lstm(param, state, input)
    weight,bias = param
    hidden,cell = state
    h       = size(hidden,2)
    gates   = hcat(input,hidden) * weight .+ bias
    forget  = sigm(gates[:,1:h])
    ingate  = sigm(gates[:,1+h:2h])
    outgate = sigm(gates[:,1+2h:3h])
    change  = tanh(gates[:,1+3h:4h])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
