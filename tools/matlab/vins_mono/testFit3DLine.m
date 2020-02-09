function testFit3DLine()
% https://www.mathworks.com/matlabcentral/answers/424591-3d-best-fit-line
% https://www.mathworks.com/matlabcentral/answers/406619-3d-coordinates-line-of-fit
% Fake n-points in R3, n=4 in your case
n = 10;
a = randn(3,1);
b = randn(3,1);
t = rand(1,10);
xyz = a + b.*t;
xyz = xyz + 0.05*randn(size(xyz)); % size 3 x n
% Engine
xzyl = fit3DLine(xyz);

% Check
x = xyz(1,:);
y = xyz(2,:);
z = xyz(3,:);
xl = xzyl(1,:);
yl = xzyl(2,:);
zl = xzyl(3,:);
close all;
figure;
plot3(x,y,z,'o'); hold on;
plot3(xl,yl,zl,'r');
axis equal; grid on;
xlabel('x');
ylabel('y');
zlabel('z');
end
