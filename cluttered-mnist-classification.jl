for p in ("Knet","ArgParse","GZip","AutoGrad","GZip","Compat", "Images", "ImageMagick", "HDF5")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse
using AutoGrad
using GZip
using Compat
using Images
using ImageMagick
using HDF5

function main(args)
    s = ArgParseSettings()
    @add_arg_table s begin
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

    read_attn_mode = false

    if o[:attention] == "true"
      read_attn_mode = true
    end

    A=100 #image width
    B=100 #image height
    img_size = B*A #canvas size
    enc_size = 256 #number of hidden units / output size in LSTM
    read_n = 12 #read glimpse grid width/height
    z_size = 10 #QSampler output size
    T = 10 #MNIST generation sequence length
    batch_size = 64 #training minibatch size
    train_iters = 10000
    learning_rate = 1e-3 #learning rate for optimizer
    bt1 = 0.5

    w = initweights(read_attn_mode, atype, A, B, enc_size)
    parameters = initparams(w, learning_rate, bt1)

    for epoch in 1:25
      (xtrnraw, ytrnraw) = loadtrainingdataset(epoch) #loads ith cluttered-mnist training dataset

      xtrn = convert(Array{Float32}, reshape(xtrnraw, B*A, div(length(xtrnraw), img_size)))
      ytrn = ytrnraw'

      dtrn = minibatch(xtrn, ytrn, batch_size)

      initialstate = [convert_if_gpu(atype, zeros(batch_size,enc_size)), convert_if_gpu(atype, zeros(batch_size,enc_size))]

      if epoch == 1
        x = convert_if_gpu(atype, dtrn[1][1])
        labels = convert_if_gpu(atype, dtrn[1][2])
        println("Epoch: ", 0, ", Loss: ", total_loss(w, batch_size, enc_size, img_size, z_size, T, x, initialstate, read_attn_mode, atype, labels, A, B, read_n))
      end

      #cum_loss = 0 #delete this line
      #mini_count = 0 #delete this line
      #correct = 0 #delete this line
      for (x,y) in dtrn #train the model
        x = convert_if_gpu(atype, x)
        y = convert_if_gpu(atype, y)

        train(w, x, parameters, batch_size, enc_size, img_size, z_size, T, initialstate, read_attn_mode, atype, y, A, B, read_n)
        #cum_loss = cum_loss + total_loss(w, batch_size, enc_size, img_size, z_size, T, x, initialstate, read_attn_mode, atype, y, A, B, read_n) #delete this line
        #mini_count = mini_count + 1 #delete this line
        #correct = correct + accuracy(y', ypred) #delete this line
        #println("Current minibatch accuracy: ", correct/(mini_count*batch_size)) #delete this line
      end

      train_cum_loss = 0
      train_mini_count = 0
      train_correct = 0
      for (x,y) in dtrn #calculate loss and accuracy for training dataset
        x = convert_if_gpu(atype, x)
        y = convert_if_gpu(atype, y)
        train_cum_loss = train_cum_loss + total_loss(w, batch_size, enc_size, img_size, z_size, T, x, initialstate, read_attn_mode, atype, y, A, B, read_n)
        train_mini_count = train_mini_count + 1
        train_correct = train_correct + accuracy(y', ypred)
      end



      println("Epoch: ", epoch, ", Training Loss: ", train_cum_loss/train_mini_count, ", Training Accuracy: ", train_correct/(train_mini_count*batch_size))
    end
end

function initweights(read_attn_mode, atype, A, B, enc_size) #0.01
  weights=Dict()
  if (!read_attn_mode)
    weights = Dict([
    ("encoder_w", 0.05*randn(2*(enc_size+A*B), 4*enc_size)),
    ("encoder_b", zeros(1,4*enc_size)),
    ("hidden1_w", 0.05*randn(enc_size,A*B)),
    ("hidden1_b", zeros(1,A*B)),
    ("hidden2_w", 0.05*randn(10,A*B)),
    ("hidden2_b", zeros(10,1))
    ])
  else
    weights = Dict([
    ("read_w", 0.05*randn(enc_size,5)),
    ("read_b", zeros(1,5)),
    ("encoder_w", 0.05*randn(800,4*enc_size)),
    ("encoder_b", zeros(1, 4*enc_size)),
    ("hidden1_w", 0.05*randn(enc_size,A*B)),
    ("hidden1_b", zeros(1,A*B)),
    ("hidden2_w", 0.05*randn(10,A*B)),
    ("hidden2_b", zeros(10,1))
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

function minibatch(X, Y, bs) #pyplot
    #takes raw input (X) and gold labels (Y)
    #returns list of minibatches (x, y)
    X = X' #(50000,10000)
    Y = Y' #(50000,10)
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

function convert_if_gpu(atype, input)
  if hasgpu == true
    return convert(atype, input)
  else
    return input
  end
end

function filterbank(gx, gy, sigma2, delta, N, A, B, atype, batch_size)
    grid_i = convert_if_gpu(atype, zeros(1, N))    #grid_i = tf.reshape(tf.cast(tf.range(N), tf.Float32), [1, -1])
    for i in 1:N
      grid_i[1,i] = i-1
    end

    mu_x = gx .+ (grid_i - N / 2 - 0.5) .* delta # eq 19
    mu_y = gy .+ (grid_i - N / 2 - 0.5) .* delta # eq 20

    a = convert_if_gpu(atype, zeros(1, 1, A))
    b = convert_if_gpu(atype, zeros(1, 1, B))

    for i in 1:A
      a[1,1,i]=i-1
    end

    for i in 1:B
      b[1,1,i]=i-1
    end

    mu_x = reshape(mu_x, batch_size, N, 1)
    mu_y = reshape(mu_y, batch_size, N, 1)

    sigma2 = reshape(sigma2, batch_size, 1, 1) # sigma2 = tf.reshape(sigma2, [-1, 1, 1])

    mu_x = convert_if_gpu(Array{Float32}, mu_x)
    mu_y = convert_if_gpu(Array{Float32}, mu_y)
    sigma2 = convert_if_gpu(Array{Float32}, sigma2)
    a = convert_if_gpu(Array{Float32}, a)
    b = convert_if_gpu(Array{Float32}, b)

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

function attn_window(scope, h_dec, N, weights, A, B, atype, batch_size)
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
    Fx, Fy = filterbank(gx, gy, sigma2, delta, N, A, B, atype, batch_size) #dims: (100,5,28)
return Fx, Fy, exp(log_gamma) #==============================================================================================================================================#
end

function read_no_attn(x, x_hat, h_dec_prev, read_n, weights, A, B, atype)
  return hcat(x, x_hat)
end

function read_attn(x, x_hat, h_dec_prev, read_n, weights, A, B, atype,batch_size)
    Fx, Fy, gamma=attn_window("read", h_dec_prev, read_n, weights, A, B, atype,batch_size)
    img=filter_img(x,Fx,Fy,gamma,read_n,A,B,atype,batch_size) # batch x (read_n*read_n)
    img_hat=filter_img(x_hat,Fx,Fy,gamma,read_n,A,B,atype,batch_size)
    return hcat(img, img_hat) # concat along feature axis
end

function filter_img(x,Fx,Fy,gamma,N,A,B,atype,batch_size)
        Fxt=permutedims(Fx, [1,3,2]) #dims(100,28,5)
        img=reshape(x,batch_size,A,B)
        Fxt=convert_if_gpu(Array{Float32}, Fxt)
        img=convert_if_gpu(Array{Float32}, img)
        Fy=convert_if_gpu(Array{Float32}, Fy)
        FxtArray = Any[]
        imgArray = Any[]
        FyArray = Any[]
        for i in 1:batch_size
          push!(FxtArray, convert_if_gpu(atype, Fxt[i,:,:]))
          push!(imgArray, convert_if_gpu(atype, img[i,:,:]))
          push!(FyArray, convert_if_gpu(atype, Fy[i,:,:]))
        end

        temp1 = imgArray[1]*FxtArray[1]
        for i in 2:batch_size
          temp = imgArray[i]*FxtArray[i]
          temp1 = vcat(temp1, temp)
        end
        temp = reshape(temp1, A, batch_size, N)
        temp = permutedims(temp, [2, 1, 3])

        temp = convert_if_gpu(Array{Float32}, temp)
        tempArray = Any[]
        for i in 1:batch_size
          push!(tempArray, convert_if_gpu(atype, temp[i,:,:]))
        end

        glimpse1 = FyArray[1]*tempArray[1]
        for i in 2:batch_size
          foo = FyArray[i]*tempArray[i]
          glimpse1 = vcat(glimpse1, foo)
        end
        glimpse = reshape(glimpse1, N, batch_size, N)
        glimpse = permutedims(glimpse, [2,1,3])
        glimpse=reshape(glimpse,batch_size,N*N)
return glimpse.*reshape(gamma,batch_size,1)
end

function encode(weights, state, input)
  return lstm(weights, "encode", state, input)
end

function binary_crossentropy(t,o)
    eps = 1e-8 # epsilon for numerical stability
    return -(t.*log(o+eps) + (1.0-t).*log(1.0-o+eps))
end

function total_loss(w, batch_size, enc_size, img_size, z_size, T, x, initialstate, read_attn_mode, atype, labels, A, B, read_n)
  if read_attn_mode
    readmethod = read_attn
  else
    readmethod = read_no_attn
  end
  h_enc_prev = convert_if_gpu(atype, zeros(batch_size, enc_size))
  enc_state = initialstate
  cs = Any[]
  for t in 1:T
    c_prev = 0
    if (t==1)
        c_prev = convert_if_gpu(atype, zeros(batch_size, img_size))
      else
        c_prev = cs[t-1]
      end
      x_hat=x-sigm(c_prev) # error image
      r = readmethod(x,x_hat,h_enc_prev, read_n, w, A, B, atype,batch_size)
      h_enc, enc_state=encode(w, enc_state, hcat(r,h_enc_prev))
      push!(cs, linear(h_enc, "hidden1", w))
      h_enc_prev=h_enc
  end
    data=sigm(cs[end])
    return softmax_cost(w,data,labels)
end

lossgradient = grad(total_loss)

function softmax_forw(weights, data)
	#applies affine transformation and softmax function
	#returns predicted probabilities
	W = weights["hidden2_w"]
  b = weights["hidden2_b"]

	### step 3
	affine = (W * data) .+ b
	affine = affine .- maximum(affine, 1)
        prob = exp(affine) ./ sum(exp(affine), 1)
        global ypred = prob
	return prob
	### step 3
end

function softmax_cost(w, data, labels)
	#takes W, b paremeters, data and correct labels
	#calculates the soft loss, gradient of w and gradient of b
  labels = labels'
  data = data'
	#start of step 3
	prob = softmax_forw(w, data)
	cost = -sum(labels .* log(prob))
	return cost
end

function accuracy(ygold, yhat)
  ygold = convert(Array{Float32}, ygold)
  yhat = convert(Array{Float32}, yhat)
	correct = 0.0
	for i=1:size(ygold, 2)
		correct += indmax(ygold[:,i]) == indmax(yhat[:, i]) ? 1.0 : 0.0
	end
	return correct
end

function train(w, x, parameters, batch_size, enc_size, img_size, z_size, T, initialstate, read_attn_mode, atype, labels, A, B, read_n)
      g = lossgradient(w, batch_size, enc_size, img_size, z_size, T, x, initialstate, read_attn_mode, atype, labels, A, B, read_n)
      for i in keys(w)
          update!(w[i], g[i], parameters[i])
      end
end

function lstm(weights, mode, state, input)

    weight = weights["encoder_w"]
    bias = weights["encoder_b"]

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

function loadtrainingdataset(i)
	info(string("Loading ", i, "th Cluttered-MNIST training dataset..."))
    xtrn = h5open(string("train_", i, ".h5"), "r") do file
      read(file, "x")
    end
    ytrn = h5open(string("train_", i, ".h5"), "r") do file
      read(file, "y")
    end
	return xtrn[:,:,17:end], ytrn[17:end,:]
end



!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
