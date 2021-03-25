function omega = angularRate(t, qwxyz)
% t: time vector in seconds
% qwxyz: quaternion array Nx4
omega = zeros(size(qwxyz, 1) - 1, 3);
dt = diff(t);
for i=1:size(qwxyz,1) - 1
    omega(i,:)= unskew(rotmat(quaternion(qwxyz(i, :)), 'point')' * ...
        rotmat(quaternion(qwxyz(i+1, :)), 'point')-eye(3))' / dt(i);
end
end
