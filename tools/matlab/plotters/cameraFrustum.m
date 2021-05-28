function [frustum3d, axes3d] = cameraFrustum(WIDTH, HEIGHT, FOCUS, scale)
% camera frame right down forward
% frustum3d 3xn each col x y z
% axes3d 6x3 each col x y z u v w
% FOCUS = 2.5
% WIDTH = 4.0
% HEIGHT = 2.25

ABCDE = [0, 0, 0;
    -WIDTH / 2.0, HEIGHT / 2.0, FOCUS;
    WIDTH / 2.0, HEIGHT / 2.0, FOCUS;
    WIDTH / 2.0, -HEIGHT / 2.0, FOCUS;
    -WIDTH / 2.0, -HEIGHT / 2.0, FOCUS]';
APEX_ORDER = 'ABCDEACDAEB';
frustum3d = zeros(3, length(APEX_ORDER));
for index = 1:length(APEX_ORDER)
    APEX_ORDER_NUM = int32(APEX_ORDER(index)) - int32('A') + 1;
    frustum3d(:, index) = ABCDE(1:3, APEX_ORDER_NUM) * scale;
end
axes3d = [0, 0, 0, 2 * FOCUS, 0, 0;
    0, 0, 0, 0, 2 * FOCUS, 0;
    0, 0, 0, 0, 0, 2 * FOCUS]' * scale;

end