function drawCameraFrustum(t_WC, R_WC, w, h, f, scale)
[frustumW, axesW] = cameraFrustumInWorld(t_WC, R_WC, w, h, f, scale);
plot3(frustumW(1, :), frustumW(2, :), frustumW(3, :), '-r', 'LineWidth', 1);
colors = {'r', 'g', 'b'};
for j = 1:3
quiver3(axesW(1, j), axesW(2, j), axesW(3, j), ...
    axesW(4, j), axesW(5, j), axesW(6, j), colors{j}, 'MaxHeadSize', 2);
end
end