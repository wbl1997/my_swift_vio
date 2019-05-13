%TEST_UMEYAMA Simple script to test the umeyama implementation
%
% This script creates a homogenous point cloud and transforms it randomly.
% The umeyama function is used to find a back transformation.
% The script can be used to test the performance of the method.
%
% Author: Christoph Graumann, 2015
%   Chair for Computer Aided Medical Procedures and Augmented Reality
%   Technische Universitaet Muenchen (Munich, Germany) and 
%   Johns Hopkins University (Baltimore, MD, USA)

%% Clear
close all;
clear;
clc;

%% Definitions
gridSize = 1 + 2*randi([10 100]); % random odd number

%% Generate surface
range = -gridSize:2:gridSize;
[X,Y] = meshgrid(range,range);
Z =  -(0.003.*X.^2 + 0.003.*Y.^2)+gridSize/2;
points = [X(:),Y(:),Z(:)]';

%% Transform surface
q= rand(4,1);
q =q/norm(q);

R = rotqr2ro(q); %rotz(randi(360))*roty(randi(360))*rotx(randi(360));
t = [randi(1000); randi(1000); randi(1000)];
s= 0.5;
points_trans = s* R*points + repmat(t,1,size(points,2));

%% Test Umeyama
[R_res, t_res] = umeyama(points,points_trans,true);

R_res
t_res
R
t

[R2, t2, s2] = ralign(points,points_trans);
R2
t2
s2