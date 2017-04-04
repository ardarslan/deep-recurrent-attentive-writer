for p in ("Knet","ArgParse","GZip","Autograd","GZip","Compat", "Images", "ImageMagick")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse
using AutoGrad
using GZip
using Compat
using Images
using ImageMagick

function main(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--outdir"; default=nothing)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    outdir = ""
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
        outdir = o[:outdir]
        !isdir(o[:outdir]) && mkdir(o[:outdir])
    end

    read_attn = false
    write_attn = false

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
    T = 10 #MNIST generation sequence length
    batch_size = 100 #training minibatch size
    train_iters = 10000
    learning_rate = 1e-3 #learning rate for optimizer
    bt1 = 0.5

    w = initweights(read_attn)
    parameters = initparams(w, learning_rate, bt1)

    (xtrnraw,ytrnraw,xtstraw,ytstraw)=loaddata()

    xtrn = convert(Array{Float32}, reshape(xtrnraw ./ 255, 28*28, div(length(xtrnraw), 784)))
    ytrnraw[ytrnraw.==0]=10;
    ytrn = convert(Array{Float32}, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)))

    xtst = convert(Array{Float32}, reshape(xtstraw ./ 255, 28*28, div(length(xtstraw), 784)))
    ytstraw[ytstraw.==0]=10;
    ytst = convert(Array{Float32}, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)))
    # seperate it into batches.

    dtrn = minibatch(xtrn, ytrn, batch_size)
    dtst = minibatch(xtst, ytst, batch_size)

    # helper function to see how your training goes on.
    #report(epoch)=println(:epoch,epoch,:trn,total_loss(w, batch_size, enc_size, dec_size, img_size, z_size, T, dtrn[1][1]))

    # report(0)
    # Main part of our training process
    #=
    @time for epoch=1:10000
        train(w, dtrn, parameters, batch_size, enc_size, dec_size, img_size, z_size, T)
        report(epoch)
    end
    =#
    x = dtrn[1][1]
    cs = Any[]
    initialstate = [zeros(batch_size,enc_size), zeros(batch_size,enc_size)]

    println("Epoch: ", 0, ", Loss: ", total_loss(w, batch_size, enc_size, dec_size, img_size, z_size, T, x, initialstate, outdir, cs))
    for t in 1:T
        out = min(1,max(0,((cs[t])'+1)/2))
        png = makegrid(out)
        filename = @sprintf("%05d_%02d.png",0,t)

        save(joinpath(outdir,filename), png)
    end
    println("INFO: 10 images were generated at the directory ", outdir)

    for epoch = 1:10000
            shuffle!(dtrn)
            index = 1
            if rem(epoch,600) == 0
              index = 1
            else
              index = rem(epoch, 600)
            end
            x = dtrn[index][1]
            cs = Any[]
            train(w, x, parameters, batch_size, enc_size, dec_size, img_size, z_size, T, initialstate, outdir, cs)
            if (rem(epoch, 10)==0)
              println("Epoch: ", epoch, ", Loss: ", total_loss(w, batch_size, enc_size, dec_size, img_size, z_size, T, x, initialstate, outdir, cs))
              for t in 1:T
                  out = min(1,max(0,((cs[t])'+1)/2))
                  png = makegrid(out)
                  filename = @sprintf("%05d_%02d.png",epoch,t)
                  save(joinpath(outdir,filename), png)
              end
              println("INFO: 10 images were generated at the directory ", outdir)
            end
    end

  #=  if o[:outdir] != nothing
        out = generate(w)
        png = makegrid(out; gridsize=[8,8], scale=2)
        filename = @sprintf("%04d.png",0)
        save(joinpath(o[:outdir],filename), png)
    end

    for epoch = 1:10
        shuffle!(dtrn)
        loss1 = test(w,dtrn)
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

    =#
end

function initweights(read_attn) #0.01
  weights=Dict()
  if (!read_attn)
    weights = Dict([
    ("encoder_w", 0.05*randn(2080, 1024)),
    ("encoder_b", zeros(1,1024)),
    ("mu_w", 0.05*randn(256,10)),
    ("mu_b", zeros(1,10)),
    ("sigma_w", 0.05*randn(256,10)),
    ("sigma_b", zeros(1,10)),
    ("decoder_w", 0.05*randn(266,1024)),
    ("decoder_b", zeros(1,1024)),
    ("write_w", 0.05*randn(256,784)),
    ("write_b", zeros(1,784))
    ])
  else
    weights = Dict([
    ("read_w", 0.05*randn(256,5)),
    ("read_b", zeros(1,5)),
    ("encoder_w", 0.05*randn(562,1024)),
    ("encoder_b", zeros(1, 1024)),
    ("mu_w", 0.05*randn(256,10)),
    ("mu_b", zeros(1,10)),
    ("sigma_w", 0.05*randn(256,10)),
    ("sigma_b", zeros(1,10)),
    ("decoder_w", 0.05*randn(266,1024)),
    ("decoder_b", zeros(1,1024)),
    ("writew_w", 0.05*randn(256,25)),
    ("writew_b", zeros(1,25)),
    ("write_w", 0.05*randn(256,5)),
    ("write_b", zeros(1,5)),
    ])
  end
  return weights
end

function initparams(w,lr,bt1)
   result = Dict()
   for k in keys(w)
     result[k] = Adam(;lr=lr, gclip=5, beta1=bt1)
   end
   return result
end

function minibatch(X, Y, bs) #pyplot
    #takes raw input (X) and gold labels (Y)
    #returns list of minibatches (x, y)
    X = X'
    Y = Y'
    data = Any[]
    for i=1:bs:size(X, 1)
	     bl = i + bs - 1 <= size(X, 1) ? i + bs - 1 : size(X, 1)
	     push!(data, (X[i:bl, :], Y[i:bl, :]))
    end
    return data
end

function linear(x, scope, weights) ##########todo
  #=
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
  =#
  w = weights[string(scope, "_w")]
  b = weights[string(scope, "_b")]
  return x*w.+b
end

function read_no_attn(x, x_hat, h_dec_prev)
  return hcat(x, x_hat)
end

readmethod = read_no_attn

function encode(weights, state, input)
  return lstm(weights, "encode", state, input)
end

function sampleQ(h_enc, weights, noise)
  mu = linear(h_enc, "mu", weights)
  logsigma = linear(h_enc, "sigma", weights)
  sigma = exp(logsigma)
  return mu+sigma.*noise, mu, logsigma, sigma
end

function decode(weights, state, input)
  return lstm(weights, "decode", state, input)
end

function write_no_attn(h_dec, weights)
  return linear(h_dec, "write", weights)
end

writemethod = write_no_attn

function binary_crossentropy(t,o)
    eps = 1e-8 # epsilon for numerical stability
    return -(t.*log(o+eps) + (1.0-t).*log(1.0-o+eps))
end

function total_loss(w, batch_size, enc_size, dec_size, img_size, z_size, T, x, initialstate, outdir, cs)

  mus = Any[]
  logsigmas = Any[]
  sigmas = Any[]

  h_dec_prev = zeros(batch_size, dec_size)

  enc_state = initialstate ##########todo

  dec_state = initialstate ##########todo

  noise = 0.05*randn(batch_size, z_size)

  #println("logsigmas: ", size(logsigmas[1]))
  for t in 1:T
    c_prev = 0
    if (t==1)
        c_prev = zeros(batch_size, img_size)
      else
        c_prev = cs[t-1]
      end
      x_hat=x-sigm(c_prev) # error image
      r = readmethod(x,x_hat,h_dec_prev)
      h_enc, enc_state=encode(w, enc_state, hcat(r,h_dec_prev))
      z,mustemp,logsigmastemp,sigmastemp=sampleQ(h_enc, w, noise)
      push!(mus, mustemp)
      push!(logsigmas, logsigmastemp)
      push!(sigmas, sigmastemp)
      h_dec,dec_state=decode(w, dec_state,z)
      cstemp=c_prev+writemethod(h_dec, w) # store results
      push!(cs, cstemp)
      h_dec_prev=h_dec
  end
    x_recons=sigm(cs[end])
    # after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
    Lx=sum(binary_crossentropy(x,x_recons),2) # reconstruction term
    Lx=mean(Lx)
    kl_terms=Any[]
    for t in 1:T
      mu2=mus[t].*mus[t]
      sigma2=sigmas[t].*sigmas[t]
      logsigma=logsigmas[t]
      push!(kl_terms, 0.5*sum((mu2+sigma2-2*logsigma),2)-T*.5) # each kl term is (1xminibatch)
    end
    KL=sum(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T ****(add_n in python = sum(x,1) in julia)
    #println("kl_terms_t: ", size(kl_terms[1]))
    #println("KL: ", size(KL))
    Lz=mean(KL) # average over minibatches                                               ****(reduce_sum(x,1) in python = sum(x,2) in julia)
    cost=Lx+Lz
    return cost
end

lossgradient = grad(total_loss)

function train(w, x, parameters, batch_size, enc_size, dec_size, img_size, z_size, T, initialstate, outdir, cs)
      g = lossgradient(w, batch_size, enc_size, dec_size, img_size, z_size, T, x, initialstate, outdir, cs)
      for i in keys(w)
          update!(w[i], g[i], parameters[i])
      end
end

function lstm(weights, mode, state, input)
  if (mode == "encode")
    weight = weights["encoder_w"]
    bias = weights["encoder_b"]
  elseif (mode == "decode")
    weight = weights["decoder_w"]
    bias = weights["decoder_b"]
  end
    cell,hidden = state
    h       = size(hidden,2)
    gates   = hcat(input,hidden) * weight .+ bias
    forget  = sigm(gates[:,1:h])
    ingate  = sigm(gates[:,1+h:2h])
    outgate = sigm(gates[:,1+2h:3h])
    change  = tanh(gates[:,1+3h:4h])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden, [cell,hidden]) ##changed
end

function makegrid(y; gridsize=[10,10], scale=2.0, shape=(28,28))
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

function loaddata()
	info("Loading MNIST...")
	xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
	xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
	ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
	ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
	return (xtrn, ytrn, xtst, ytst)
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
