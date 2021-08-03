function [frustum3dW, axes3dW] = cameraFrustumInWorld(t_WC, R_WC, w, h, f, scale)
% t_WC 3x1
% R_WC 3x3
% frustum3d 3xn each col x y z
% axes3d 6x3 each col x y z u v w
[frustum3d, axes3d] = cameraFrustum(w, h, f, scale);
frustum3dW = R_WC * frustum3d + repmat(t_WC, 1, size(frustum3d, 2));

axes3dW = [R_WC * axes3d(1:3, :) + repmat(t_WC, 1, 3); ...
    R_WC * axes3d(4:6, :)];
end
