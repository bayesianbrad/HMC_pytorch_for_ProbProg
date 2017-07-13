% EXAMPLE 2: HYBRID MONTE CARLO SAMPLING -- BIVARIATE NORMAL
rand('seed',12345);
randn('seed',12345);
 
% STEP SIZE
epsilon = 0.18;
nSamples = 1000;
L = 20;
 
% DEFINE POTENTIAL ENERGY FUNCTION
U = inline('transp(q)*inv([1,.98;.98,1])*q','q');
 
% DEFINE GRADIENT OF POTENTIAL ENERGY
dU = inline('transp(q)*inv([1,.98;.98,1])','q');
 
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
 
% DISPLAY
figure
scatter(q(1,:),q(2,:),'k.'); hold on;
plot(q(1,1:50),q(2,1:50),'ro-','Linewidth',2);
xlim([-6 6]); ylim([-6 6]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Hamiltonian Monte Carlo')
