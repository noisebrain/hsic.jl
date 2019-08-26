using LinearAlgebra
using Distances

# unittest performance
# versus python hsic_varioustests_npy.py unittest:
#    20sec elapsed / 52sec user (i.e. multithreaded)
# using distmat1: 10.3sec
# using Distances.jl pairwise 13.2sec
# using distmatvec: 10.7sec

# performance discussion see https://discourse.julialang.org/t/sum-of-hadamard-products/3531/5

# for comparison, do the non-vectorized calculation
# points are in columns, dimensions(variables) across rows
# note transposed from previous convention!
function distmatslow(X,Y)
  m = size(X,2)
  @assert m==size(Y,2) #"size mismatch"
  D = Array{Float32}(undef,m,m)
  for ix = 1:m
    @inbounds for iy = 1:m
      x = X[:,ix]
      y = Y[:,iy]
      d = norm(x-y)
      D[ix,iy] = d*d
    end
  end
  return D
end

function distmat(X,Y)
  m = size(X,2)
  @assert m==size(Y,2) #"size mismatch"
  D = Array{Float32}(undef,m,m)
  for ix = 1:m
    @inbounds for iy = ix:m
      x = X[:,ix]
      y = Y[:,iy]
      d = x-y
      d2 = d'*d
      D[ix,iy] = d2
      D[iy,ix] = d2
    end
  end
  return D
end

#
function distmatvec(X,Y)
  """
  points are in columns, dimensions(variables) down rows
  """
  m = size(X,2)
  @assert m==size(Y,2) #"size mismatch"

  XY = sum(X .* Y,dims=1)
  #XY = XY.reshape(m,1)
  #R1 = n_.tile(XY,(1,Y.shape[0]))
  R1 = repeat(XY',1,m)  # outer=(1,m) not suported by autograd
  R2 = repeat(XY,m,1)
  #xy_ = X * Y'
  D = R1 + R2 - 2.f0 * X' * Y
  D
end

# python-ish version requires broadcasting different shapes - "outer sum"
function distmat1(X,Y)
  A = sum(X .* X,dims=1)'
  B = sum(Y .* Y,dims=1)
  C = X' * Y
  A .+ B .- 2.f0 * C
end

function eye(n)
  Matrix{Float32}(I,n,n)
end

function hsic(X,Y,sigma)
  #=
  # 1/m^2 Tr Kx H Ky H
  X,Y have data in COLUMNS, each row is a dimension
  # NB is transposed from python
  =#
  #println("X=\n", X'[1:2,:])
  #Yt = Y;; println("Y=\n", Yt[1:2:])
  m = size(X,2)
  println("hsic between ",m,"points")
  H = eye(m) - (1.f0 / m) * ones(Float32,m,m)
  #Dxx = distmatvec(X,X)
  #Dyy = distmatvec(Y,Y)
  Dxx = distmat1(X,X)
  Dyy = distmat1(Y,Y)
  #Dxx = pairwise(SqEuclidean(),X,X,dims=2)
  #Dyy = pairwise(SqEuclidean(),Y,Y,dims=2)
  sigma2 = 2.f0 * sigma*sigma
  Kx = exp.( -Dxx / sigma2 )
  Ky = exp.( -Dyy / sigma2 )
  Kxc = Kx * H
  Kyc = Ky * H
  thehsic = (1.f0 / (m*m)) * sum(Kxc' .* Kyc)
  return thehsic	# type float32
end

# Pkg.add("NPZ")
using NPZ
# aug19: this matches the output of the unittest in hsic_varioustests_npy
function unittest()
  X = Float32[0.1 0.2 0.3;
                5 4 3]
  Y = Float32[1 2 3;
               2 2 2]
  X = transpose(X)
  Y = transpose(Y)
  println(distmatslow(X,X))
  println(distmat(X,X))
  println(distmatslow(Y,Y))
  println(distmat(Y,Y))
  println("hsic(X,Y,0.5)=",hsic(X,Y,0.5f0))

  # larger test
  data = npzread("/tmp/_data.npz")
  # todo convert arrays to float32
  X = data["arr_0"]
  Y = data["arr_1"]
  X = convert(Array{Float32,2},X')
  Y = convert(Array{Float32,2},Y')
  println("X=\n", X'[1:2,:])
  println("Y=\n", Y'[1:2,:])

  println("\nindependent hsic(X,Y,1)=",hsic(X,Y,1.f0))

  println("\nidentical hsic(X,X,1)=",hsic(X,copy(X),1.f0))

  Y2 = X .* X
  println("Y2=",Y2'[1:2,:])
  println("\nnonlinear hsic(X,Y*Y,1)=",hsic(X,Y2,1.f0))
end
