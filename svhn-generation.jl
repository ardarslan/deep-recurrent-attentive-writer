for p in ("Knet","ArgParse","GZip","AutoGrad","GZip","Compat", "Images","ImageMagick", "MAT")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse
using AutoGrad
using GZip
using Compat
using Images
using MAT
using ImageMagick

function main(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--outdir"; default=nothing)
        ("--attention"; default="false")
    end

    global hasgpu = false
    atype = Array{Float32}
    if gpu()>=0
      atype = KnetArray{Float32}
      global hasgpu = true
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    global outdir = ""
    read_attn_mode = false
    write_attn_mode = false

    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
        global outdir = o[:outdir]
        !isdir(o[:outdir]) && mkdir(o[:outdir])
    end
    if o[:attention] == "true"
      read_attn_mode = true
      write_attn_mode = true
    end

    A=32 #image width
    B=32 #image height
    num_colors = 3
    n_hidden = 800 #number of hidden units / output size in LSTM
    read_n = 12 #read glimpse grid width/height
    write_n = 12 #write glimpse grid width/height
    z_size = 10 #QSampler output size
    T = 32 #MNIST generation sequence length
    batch_size = 100 #training minibatch size
    train_iters = 10000
    learning_rate = 1e-3 #learning rate for optimizer
    bt1 = 0.5

    w = initweights(read_attn_mode, atype, A, B, n_hidden, num_colors, read_n, write_n)
    parameters = initparams(w, learning_rate, bt1)

    xtrnraw, xtstraw=loaddata()
    xtrn = convert(Array{Float32}, reshape(xtrnraw ./ 255, A*B*num_colors, div(length(xtrnraw), A*B*num_colors))) #dims:(3072,73200)
    xtst = convert(Array{Float32}, reshape(xtstraw ./ 255, A*B*num_colors, div(length(xtstraw), A*B*num_colors))) #dims:(3072,73200)
    # seperate it into batches.


    dtrn = minibatch(xtrn, batch_size)
    dtst = minibatch(xtst, batch_size)

    #=
    println(size((dtrn[224,1])'))
    out = (dtrn[224,1])'
    png = makegrid(out, scale=1.0, shape=(A,B))
    filename = @sprintf("2420%05d_%02d.png",0,0)
    save(joinpath(outdir,filename), png)
    =#


    x = convert_if_gpu(atype, dtrn[1])  #dims:(100,3072)
    cs = Any[]
    attn_params = Any[]
    initialstate = [convert_if_gpu(atype, zeros(batch_size,n_hidden)), convert_if_gpu(atype, zeros(batch_size,n_hidden))]

    println("Epoch: ", 0, ", Loss: ", total_loss(w, batch_size, n_hidden, B*A, z_size, T, x, initialstate, outdir, cs, read_attn_mode, write_attn_mode, A, B, num_colors,atype,read_n,write_n))

    for t in 1:T
        out = min(1,max(0,((cs[t])'+1)/2)) #out = min(1,max(0,((cs[t])'+1)/2))
        png = makegrid(out, scale=1.0, shape=(A,B))
        filename = @sprintf("%05d_%02d.png",0,t)
        save(joinpath(outdir,filename), png)
    end
    println("INFO: 10 images were generated at the directory ", outdir)




        #load("data.jld")["data"]

        #firstimage = Images.colorview(RGB, y[:,:,:,1])
        #outdir = "/ec2-user/svhngenerations"
        #filename = string("epoch", epoch, "instance1.png")
        #save(joinpath(outdir,filename), firstImage)

        #makegrid(out,0,scale=1.0, shape=(A,B))

        #filename = @sprintf("%05d_%02d.png",0,t)

        #save(joinpath(outdir,filename), png)

    #println("INFO: 10 images were generated at the directory ", outdir)


    for epoch = 1:10000
            indextrn = 1
            if rem(epoch,732) == 0
              indextrn = 1
            else
              indextrn = rem(epoch, 732)
            end

            indextst = 1
            if rem(epoch,260) == 0
              indextst = 1
            else
              indextst = rem(epoch, 260)
            end

            x = dtrn[indextrn]
            cs1 = Any[]
            cs2 = Any[]
            train(w, x, parameters, batch_size, n_hidden, B*A, z_size, T, initialstate, outdir, cs1, read_attn_mode, write_attn_mode, A, B, num_colors,atype,read_n,write_n)

            if (rem(epoch, 10)==0)
              train_loss = total_loss(w, batch_size, n_hidden, B*A, z_size, T, x, initialstate, outdir, cs1, read_attn_mode, write_attn_mode, A, B, num_colors,atype,read_n,write_n)
              x = dtst[indextst]
              test_loss = total_loss(w, batch_size, n_hidden, B*A, z_size, T, x, initialstate, outdir, cs2, read_attn_mode, write_attn_mode, A, B, num_colors,atype,read_n,write_n)



              for t in 1:T
                  out = min(1,max(0,((cs1[t])'+1)/2)) #out = min(1,max(0,((cs[t])'+1)/2))
                  png = makegrid(out, scale=1.0, shape=(A,B))
                  filename = @sprintf("trn_%05d_%02d.png",epoch,t)
                  save(joinpath(outdir,filename), png)
              end

              for t in 1:T
                  out = min(1,max(0,((cs2[t])'+1)/2)) #out = min(1,max(0,((cs[t])'+1)/2))
                  png = makegrid(out, scale=1.0, shape=(A,B))
                  filename = @sprintf("tst_%05d_%02d.png",epoch,t)
                  save(joinpath(outdir,filename), png)
              end

              println("Epoch: ", epoch, ", TrnLoss: ", train_loss, ", TstLoss: ", test_loss)
              println("INFO: 64 images were generated at the directory ", outdir)



                  #filename = @sprintf("%05d_%02d.png",0,t)

                  #save(joinpath(outdir,filename), png)

              #println("INFO: 10 images were generated at the directory ", outdir)

            end
    end
end

function initweights(read_attn_mode, atype, A, B, n_hidden, num_colors, read_n, write_n) #0.01
  weights=Dict()
  if (!read_attn_mode)
    weights = Dict([
    ("encoder_w", randn(7744, n_hidden*4)), #Use 6656 when enc_size=256
    ("encoder_b", zeros(1,n_hidden*4)),
    ("mu_w", 0.05*randn(n_hidden,10)),
    ("mu_b", zeros(1,10)),
    ("sigma_w", 0.05*randn(n_hidden,10)),
    ("sigma_b", zeros(1,10)),
    ("decoder_w", 0.05*randn(n_hidden+10,n_hidden*4)),
    ("decoder_b", zeros(1,n_hidden*4)),
    ("write_w", 0.05*randn(n_hidden,A*B*num_colors)),
    ("write_b", zeros(1,A*B*num_colors))
    ])
  else
    weights = Dict([
    ("read_w", 0.05*randn(n_hidden,5)),
    ("read_b", zeros(1,5)),
    ("encoder_w", 0.05*randn(2464,n_hidden*4)),  #Use 1376 when enc_size=256
    ("encoder_b", zeros(1, n_hidden*4)),
    ("mu_w", 0.05*randn(n_hidden,10)),
    ("mu_b", zeros(1,10)),
    ("sigma_w", 0.05*randn(n_hidden,10)),
    ("sigma_b", zeros(1,10)),
    ("decoder_w", 0.05*randn(n_hidden+10,n_hidden*4)),
    ("decoder_b", zeros(1,n_hidden*4)),
    ("writeW_w", 0.05*randn(n_hidden,write_n*write_n*num_colors)),
    ("writeW_b", zeros(1,write_n*write_n*num_colors)),
    ("write_w", 0.05*randn(n_hidden,write_n)),
    ("write_b", zeros(1,write_n)),
    ])
  end
  weightsKnet = Dict()
  for k in keys(weights)
    weightsKnet[k] = convert_if_gpu(atype, weights[k])
  end
  return weightsKnet
end

function initparams(w,lr,bt1)
   result = Dict()
   for k in keys(w)
     result[k] = Adam(;lr=lr, gclip=5, beta1=bt1)
   end
   return result
end

function minibatch(X, bs) #pyplot
    #takes raw input (X) and gold labels (Y)
    #returns list of minibatches (x, y)
    X = X'
    data = Any[]
    for i=1:bs:size(X, 1)
	     bl = i + bs - 1 <= size(X, 1) ? i + bs - 1 : size(X, 1)
	     push!(data, X[i:bl, :])
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

function convert_if_gpu(atype, input)
  if hasgpu == true
    return convert(atype, input)
  else
    return input
  end
end


function filterbank(gx, gy, sigma2, delta, N, A, B, atype)

    grid_i = convert_if_gpu(atype, zeros(1, N))    #grid_i = tf.reshape(tf.cast(tf.range(N), tf.Float32), [1, -1])
    for i in 1:N
      grid_i[1,i] = i-1
    end

    mu_x = gx .+ (grid_i - N / 2 - 0.5) .* delta # eq 19
    mu_y = gy .+ (grid_i - N / 2 - 0.5) .* delta # eq 20

    a = zeros(1, 1, A)
    b = zeros(1, 1, B)

    for i in 1:A
      a[1,1,i]=i-1
    end

    for i in 1:B
      b[1,1,i]=i-1
    end

    mu_x = reshape(mu_x, 100, N, 1)
    mu_y = reshape(mu_y, 100, N, 1)

    sigma2 = reshape(sigma2, 100, 1, 1) # sigma2 = tf.reshape(sigma2, [-1, 1, 1])

    mu_x = convert_if_gpu(Array{Float32}, mu_x)
    mu_y = convert_if_gpu(Array{Float32}, mu_y)
    sigma2 = convert_if_gpu(Array{Float32}, sigma2)

    Fxbroad = (a.-mu_x) ./ (2*sigma2)
    Fybroad = (b.-mu_y) ./ (2*sigma2)

    Fx = exp(-Fxbroad.*Fxbroad) # 2*sigma2?
    Fy = exp(-Fybroad.*Fybroad)
    # normalize, sum over A and B dims
    Fx=Fx./max(sum(Fx,3),1e-8) #dims: (100,5,28)
    Fy=Fy./max(sum(Fy,3),1e-8) #dims: (100,5,28)

    Fx = convert_if_gpu(atype, Fx)
    Fy = convert_if_gpu(atype, Fy)
return Fx,Fy
end

function attn_window(scope, h_dec, N, weights, A, B, atype)
    params = linear(h_dec, scope, weights) #dims: (100,5)
    gx_ = params[:,1]
    gy_ = params[:,2]
    log_sigma2 = params[:,3] #dims: (100,1)
    log_delta = params[:,4]
    log_gamma = params[:,5] #dims: (100,1)
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=exp(log_sigma2) #dims: (100,1)
    delta=(max(A,B)-1)/(N-1)*exp(log_delta) # batch x N
    Fx, Fy = filterbank(gx, gy, sigma2, delta, N, A, B, atype) #dims: (100,5,28)
return Fx, Fy, exp(log_gamma) #==============================================================================================================================================#
end

function read_no_attn(x, x_hat, h_dec_prev, read_n, weights, A, B, num_colors, atype)
  return hcat(x, x_hat)
end

function read_attn(x, x_hat, h_dec_prev, read_n, weights, A, B, num_colors, atype)
    Fx, Fy, gamma=attn_window("read", h_dec_prev, read_n, weights, A, B, atype)
    img=filter_img(x,Fx,Fy,gamma,read_n,A,B,num_colors,atype) # batch x (read_n*read_n)
    img_hat=filter_img(x_hat,Fx,Fy,gamma,read_n,A,B,num_colors,atype)
    return hcat(img, img_hat) # concat along feature axis
end

function filter_img(x,Fx,Fy,gamma,N,A,B,num_colors,atype)

  x = convert_if_gpu(Array{Float32}, x)
  Fx = convert_if_gpu(Array{Float32}, Fx)
  Fy = convert_if_gpu(Array{Float32}, Fy)
  gamma = convert_if_gpu(Array{Float32}, gamma)

  img=reshape(x,100,B,A,num_colors)
  img_t = permutedims(img, [4,1,2,3])
  batch_colors_array = reshape(img_t, num_colors * 100, B, A)
  Fx_array = vcat(Fx,Fx)
  Fx_array = vcat(Fx_array,Fx)
  Fxt=permutedims(Fx_array, [1,3,2]) #dims(300,32,5)
  Fy_array = vcat(Fy,Fy)
  Fy_array = vcat(Fy_array,Fy)
  temp1 = batch_colors_array[1,:,:]*Fxt[1,:,:]
  for i in 2:300
    temp = batch_colors_array[i,:,:]*Fxt[i,:,:]
    temp1 = vcat(temp1, temp)
  end
  temp = reshape(temp1, A, 300, N)
  temp = permutedims(temp, [2, 1, 3])

  glimpse1 = Fy_array[1,:,:]*temp[1,:,:]
  for i in 2:300
    foo = Fy_array[i,:,:]*temp[i,:,:]
    glimpse1 = vcat(glimpse1, foo)
  end
  glimpse = reshape(glimpse1, N, 300, N)
  glimpse = permutedims(glimpse, [2,1,3])
  glimpse = reshape(glimpse, num_colors, 100, N, N)
  glimpse = permutedims(glimpse, [2,3,4,1])
  glimpse = reshape(glimpse, 100, N*N*3)
  result = glimpse.*reshape(gamma,100,1)
return convert_if_gpu(atype, result)
end

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

function write_no_attn(h_dec, write_n, batch_size, weights, A, B, num_colors,atype)
  return linear(h_dec, "write", weights)
end

function write_attn(h_dec, write_n, batch_size, weights, A, B, num_colors,atype)
  w = linear(h_dec, "writeW", weights)
  w = convert_if_gpu(Array{Float32}, w)
  N = write_n
  w = reshape(w, batch_size, N, N, num_colors) #dims: (100,5,5,3)
  w_t = permutedims(w, [4,1,2,3]) #dims: (3,100,5,5)
  Fx, Fy, gamma = attn_window("write", h_dec, write_n, weights, A, B, atype)
  Fx = convert_if_gpu(Array{Float32}, Fx)
  Fy = convert_if_gpu(Array{Float32}, Fy)
  gamma = convert_if_gpu(Array{Float32}, gamma)

  w_array = reshape(w_t, num_colors*batch_size, write_n, write_n) #dims: (300,5,5)
  Fx_array = vcat(Fx, Fx)
  Fx_array = vcat(Fx_array, Fx) #dims(300,5,32)
  Fy_array = vcat(Fy, Fy)
  Fy_array = vcat(Fy_array, Fy) #dims(300,5,32)

  Fyt=permutedims(Fy_array, [1,3,2]) #dims: (300,32,5)
  temp1 = w_array[1,:,:]*Fx_array[1,:,:]
  for i in 2:size(w_array)[1]
    foo = w_array[i,:,:]*Fx_array[i,:,:]
    temp1 = vcat(temp1, foo)
  end
  temp = reshape(temp1, write_n, 300, A)
  temp = permutedims(temp, [2,1,3]) #dims: (300,5,32)

  wr1 = Fyt[1,:,:]*temp[1,:,:]
  for i in 2:size(temp)[1]
    foo = Fyt[i,:,:]*temp[i,:,:]
    wr1 = vcat(wr1, foo)
  end
  wr = reshape(wr1, B, 300, A)
  wr = permutedims(wr, [2,1,3])
  wr=reshape(wr,num_colors,batch_size,B,A)
  wr=permutedims(wr,[2,3,4,1])
  wr=reshape(wr,100,A*B*num_colors)
return convert_if_gpu(atype, wr.*reshape(1 ./ gamma, 100, 1))
end

function crossentropy(t)
    t.*t/2
end

function total_loss(w, batch_size, n_hidden, img_size, z_size, T, x, initialstate, outdir, cs, read_attn_mode, write_attn_mode, A, B, num_colors,atype,read_n,write_n)
  if read_attn_mode
    readmethod = read_attn
  else
    readmethod = read_no_attn
  end

  if write_attn_mode
    writemethod = write_attn
  else
    writemethod = write_no_attn
  end

  mus = Any[]
  logsigmas = Any[]
  sigmas = Any[]

  h_dec_prev = convert_if_gpu(atype, zeros(batch_size, n_hidden))

  enc_state = initialstate ##########todo

  dec_state = initialstate ##########todo

  noise = 0.05*convert_if_gpu(atype, randn(batch_size, z_size))
  x=convert_if_gpu(atype, x)

  #println("logsigmas: ", size(logsigmas[1]))
  for t in 1:T
    c_prev = 0
    if (t==1)
        c_prev = convert_if_gpu(atype, zeros(batch_size, img_size*num_colors))
      else
        c_prev = cs[t-1]
      end
      x_hat=x-sigm(c_prev) # error image
      r = readmethod(x,x_hat,h_dec_prev, read_n, w, A, B,num_colors,atype)
      h_enc, enc_state=encode(w, enc_state, hcat(r,h_dec_prev))
      z,mustemp,logsigmastemp,sigmastemp=sampleQ(h_enc, w, noise)
      push!(mus, mustemp)
      push!(logsigmas, logsigmastemp)
      push!(sigmas, sigmastemp)
      h_dec,dec_state=decode(w, dec_state,z)
      cstemp=c_prev+writemethod(h_dec, write_n, batch_size, w, A, B,num_colors,atype) # store results
      push!(cs, cstemp)
      h_dec_prev=h_dec
  end
    x_recons=sigm(cs[end])
    # after computing cross entropy, sum across features then take the mean of those sums across minibatches
    Lx=sum(crossentropy(x-x_recons),2) # reconstruction term
    Lx=sum(Lx)/length(Lx)
    kl_terms=Any[]
    for t in 1:T
      mu2=mus[t].*mus[t]
      sigma2=sigmas[t].*sigmas[t]
      logsigma=logsigmas[t]
      push!(kl_terms, 0.5*sum((mu2+sigma2-2*logsigma),2)-z_size*.5) # each kl term is (1xminibatch)
    end
    KL=sum(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T ****(add_n in python = sum(x,1) in julia)
    #println("kl_terms_t: ", size(kl_terms[1]))
    #println("KL: ", size(KL))
    Lz=sum(KL)/length(KL) # average over minibatches                                               ****(reduce_sum(x,1) in python = sum(x,2) in julia)
    cost=Lx+Lz
    return cost
end

lossgradient = grad(total_loss)

function train(w, x, parameters, batch_size, n_hidden, img_size, z_size, T, initialstate, outdir, cs, read_attn_mode, write_attn_mode, A, B, num_colors,atype,read_n,write_n)
      g = lossgradient(w, batch_size, n_hidden, img_size, z_size, T, x, initialstate, outdir, cs, read_attn_mode, write_attn_mode, A, B, num_colors,atype,read_n,write_n)
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

function makegrid(y; gridsize=[10,10], scale=2.0, shape=(32,32))
    y = convert_if_gpu(Array{Float32}, y)
    y = reshape(y, 32,32,3,100)
    y = permutedims(y, [3,1,2,4])
    shp = map(x->Int(round(x*scale)), shape)
    t = Any[]
    for i in 1:100
      push!(t, y[:,:,:,i])
    end
    y = map(x->Images.colorview(RGB, x), t)
    gridx, gridy = gridsize
    out = zeros(3,331,331)
    out = Images.colorview(RGB, out)
    #for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:] = 1.0; end
    #for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1] = 1.0; end

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

    return out
end

function loaddata()
	info("Loading SVHN...")
	gzload("train_32x32.mat")
  gzload("test_32x32.mat")
  xtrn = matread("train_32x32.mat")["X"]
  xtrn = view(xtrn, :,:,:,58:73257)
  xtst = matread("test_32x32.mat")["X"]
  xtst = view(xtst, :,:,:,33:26032)
  return xtrn, xtst #dims:(32,32,3,73200)
end

function gzload(file; path="$file", url="http://ufldl.stanford.edu/housenumbers/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
