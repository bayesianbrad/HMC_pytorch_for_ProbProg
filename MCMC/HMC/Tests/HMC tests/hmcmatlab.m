% EXAMPLE 2: HYBRID MONTE CARLO SAMPLING -- BIVARIATE NORMAL
rand('seed',12345);
randn('seed',12345);
 
% STEP SIZE
epsilon = 0.18;
nSamples = 10000;
L = 5;
 
% DEFINE POTENTIAL ENERGY FUNCTION
U = inline('transp(q)*inv([1,0.8;0.8,1])*q','q');
 
% DEFINE GRADIENT OF POTENTIAL ENERGY
dU = inline('transp(q)*inv([1,0.8;0.8,1])','q');
 
% DEFINE KINETIC ENERGY FUNCTION
K = inline('sum((transp(p)*p))/2','p');
 
% INITIAL STATE
q = zeros(2,nSamples);
q0 = [-1.5;-1.55];
q(:,1) = q0;
 
t = 1;
while t < nSamples
    t = t + 1;
 
    % SAMPLE RANDOM MOMENTUM
    p0 = randn(2,1);
 
    %% SIMULATE HAMILTONIAN DYNAMICS
    % FIRST 1/2 STEP OF MOMENTUM
    p = p0 - epsilon/2*dU(q(:,t-1))';
 
    % FIRST FULL STEP FOR POSITION/SAMPLE
    qprop = q(:,t-1) + epsilon*p;
 
    % FULL STEPS
    for jL = 1:L-1
        % MOMENTUM
        p = p - epsilon*dU(qprop)';
        % POSITION/SAMPLE
        qprop = qprop + epsilon*p;
    end
 
    % LAST HALP STEP
    p = p - epsilon/2*dU(qprop)';
 
    % COULD NEGATE MOMENTUM HERE TO LEAVE
    % THE PROPOSAL DISTRIBUTION SYMMETRIC.
    % HOWEVER WE THROW THIS AWAY FOR NEXT
    % SAMPLE, SO IT DOESN'T MATTER
 
    % EVALUATE ENERGIES AT
    % START AND END OF TRAJECTORY
    Ucurrent = U(q(:,t-1));
    Uprop = U(qprop);
 
    Kcurrent = K(p0);
    Kprop = K(p);
 
    % ACCEPTANCE/REJECTION CRITERION
    alpha = min(1,exp((Ucurrent + Kcurrent) - (Uprop + Kprop)));
 
    u = rand;  
    if u < alpha
        q(:,t) = qprop;
    else
        q(:,t) = q(:,t-1);
    end
end
mean1  = mean(q(:,1))
mean2  = mean(q(:,2))
% DISPLAY
figure
scatter(q(1,:),q(2,:),'k.'); hold on;
plot(q(1,:),q(2,:),'ro-','Linewidth',2);
hold on
SIGMA = [1,0.8;0.8,1];
MU = [0;0]';
x = linspace(-2,2,1000);
y = linspace(-2,2,1000);
[X,Y] = meshgrid(x,y);
z = mvnpdf([X(:) Y(:)],MU,SIGMA);
z = reshape(z,size(X));
contour(X,Y,z)
xlim([-6 6]); ylim([-6 6]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Hamiltonian Monte Carlo')
