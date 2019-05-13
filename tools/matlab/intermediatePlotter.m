function figNumber = intermediatePlotter(figNumber, data, gt, outputPath)
if(nargin==3)
outputPath= 'G:\temp';
end
figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 19)-1, '-r'); hold on;
plot(data(:,1), data(:, 20), '-g');
plot(data(:,1), data(:, 21), '-b');
plot(data(:,1), data(:, 22), '-k');
plot(data(:,1), data(:, 23)-1, '.k');
plot(data(:,1), data(:, 24), '.b');
plot(data(:,1), data(:, 25), '-c');
plot(data(:,1), data(:, 26), '-m');
plot(data(:,1), data(:, 27)-1, '-y');

plot(data(:,1), 3*data(:, 74), '--r');
plot(data(:,1), 3*data(:, 75), '--g');
plot(data(:,1), 3*data(:, 76), '--b');
plot(data(:,1), 3*data(:, 77), '--k');
plot(data(:,1), 3*data(:, 78), '-.k');
plot(data(:,1), 3*data(:, 79), '-.b');
plot(data(:,1), 3*data(:, 80), '--c');
plot(data(:,1), 3*data(:, 81), '--m');
plot(data(:,1), 3*data(:, 82), '--y');

plot(data(:,1), -3*data(:, 74), '--r');
plot(data(:,1), -3*data(:, 75), '--g');
plot(data(:,1), -3*data(:, 76), '--b');
plot(data(:,1), -3*data(:, 77), '--k');
plot(data(:,1), -3*data(:, 78), '-.k');
plot(data(:,1), -3*data(:, 79), '-.b');
plot(data(:,1), -3*data(:, 80), '--c');
plot(data(:,1), -3*data(:, 81), '--m');
plot(data(:,1), -3*data(:, 82), '--y');
h_legend = legend('1','2','3','4','5','6','7','8','9',...
    '3\sigma_1','3\sigma_2','3\sigma_3','3\sigma_4','3\sigma_5','3\sigma_6','3\sigma_7','3\sigma_8','3\sigma_9');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{T}_g$' , 'Interpreter', 'Latex');
set(gca,'FontSize',18);


figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 28), '-r'); hold on;
plot(data(:,1), data(:, 29), '-g');
plot(data(:,1), data(:, 30), '-b');
plot(data(:,1), data(:, 31), '-k');
plot(data(:,1), data(:, 32), '.k');
plot(data(:,1), data(:, 33), '.b');
plot(data(:,1), data(:, 34), '-c');
plot(data(:,1), data(:, 35), '-m');
plot(data(:,1), data(:, 36), '-y');

plot(data(:,1), 3*data(:, 83), '--r');
plot(data(:,1), 3*data(:, 84), '--g');
plot(data(:,1), 3*data(:, 85), '--b');
plot(data(:,1), 3*data(:, 86), '--k');
plot(data(:,1), 3*data(:, 87), '-.k');
plot(data(:,1), 3*data(:, 88), '-.b');
plot(data(:,1), 3*data(:, 89), '--c');
plot(data(:,1), 3*data(:, 90), '--m');
plot(data(:,1), 3*data(:, 91), '--y');

plot(data(:,1), -3*data(:, 83), '--r');
plot(data(:,1), -3*data(:, 84), '--g');
plot(data(:,1), -3*data(:, 85), '--b');
plot(data(:,1), -3*data(:, 86), '--k');
plot(data(:,1), -3*data(:, 87), '-.k');
plot(data(:,1), -3*data(:, 88), '-.b');
plot(data(:,1), -3*data(:, 89), '--c');
plot(data(:,1), -3*data(:, 90), '--m');
plot(data(:,1), -3*data(:, 91), '--y');

h_legend= legend('1','2','3','4','5','6','7','8','9',...
    '3\sigma_1','3\sigma_2','3\sigma_3','3\sigma_4','3\sigma_5','3\sigma_6','3\sigma_7','3\sigma_8','3\sigma_9');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{T}_s$', 'Interpreter', 'Latex');
set(gca,'FontSize',18);

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 37)-1, '-r'); hold on;
plot(data(:,1), data(:, 38), '-g');
plot(data(:,1), data(:, 39), '-b');
plot(data(:,1), data(:, 40), '-k');
plot(data(:,1), data(:, 41)-1, '.k');
plot(data(:,1), data(:, 42), '.b');
plot(data(:,1), data(:, 43), '-c');
plot(data(:,1), data(:, 44), '-m');
plot(data(:,1), data(:, 45)-1, '-y');

plot(data(:,1), 3*data(:, 92), '--r');
plot(data(:,1), 3*data(:, 93), '--g');
plot(data(:,1), 3*data(:, 94), '--b');
plot(data(:,1), 3*data(:, 95), '--k');
plot(data(:,1), 3*data(:, 96), '-.k');
plot(data(:,1), 3*data(:, 97), '-.b');
plot(data(:,1), 3*data(:, 98), '--c');
plot(data(:,1), 3*data(:, 99), '--m');
plot(data(:,1), 3*data(:, 100), '--y');

plot(data(:,1), -3*data(:, 92), '--r');
plot(data(:,1), -3*data(:, 93), '--g');
plot(data(:,1), -3*data(:, 94), '--b');
plot(data(:,1), -3*data(:, 95), '--k');
plot(data(:,1), -3*data(:, 96), '-.k');
plot(data(:,1), -3*data(:, 97), '-.b');
plot(data(:,1), -3*data(:, 98), '--c');
plot(data(:,1), -3*data(:, 99), '--m');
plot(data(:,1), -3*data(:, 100), '--y');
h_legend=legend('1','2','3','4','5','6','7','8','9',...,
    '3\sigma_1','3\sigma_2','3\sigma_3','3\sigma_4','3\sigma_5','3\sigma_6','3\sigma_7','3\sigma_8','3\sigma_9');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{T}_a$', 'Interpreter', 'Latex');
set(gca,'FontSize',18);

figNumber = figNumber +1;
figure(figNumber);
ruler = 100;
plot(data(:,1), data(:, 46)*ruler, '-r'); hold on;
plot(data(:,1), data(:, 47)*ruler, '-g');
plot(data(:,1), data(:, 48)*ruler, '-b');

plot(data(:,1), 3*data(:, 101)*ruler, '--r');
plot(data(:,1), 3*data(:, 102)*ruler, '--g');
plot(data(:,1), 3*data(:, 103)*ruler, '--b');

plot(data(:,1), -3*data(:, 101)*ruler, '--r');
plot(data(:,1), -3*data(:, 102)*ruler, '--g');
plot(data(:,1), -3*data(:, 103)*ruler, '--b');
h_legend=legend('x','y','z', '3\sigma_x','3\sigma_y','3\sigma_z');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$\mathbf{p}_b^c$', 'Interpreter', 'Latex');
set(gca,'FontSize',18);
ylabel('cm', 'FontSize', 18);
saveas(gcf,[outputPath, '\Error p_CB'],'epsc');

figNumber = figNumber +1;
figure(figNumber);
if(size(gt,1) == size(data,1))
    plot(data(:,1), data(:, 49)-gt(:,49), '-r'); hold on;
    plot(data(:,1), data(:, 50)-gt(:,50), '-g');
    plot(data(:,1), data(:, 51)-gt(:,51), '-b');
    plot(data(:,1), data(:, 52)-gt(:,52), '-k');
else if(size(gt,1)==4)
        plot(data(:,1), data(:, 49)-gt(1), '-r'); hold on;
        plot(data(:,1), data(:, 50)-gt(2), '-g');
        plot(data(:,1), data(:, 51)-gt(3), '-b');
        plot(data(:,1), data(:, 52)-gt(4), '-k');
    end
end

plot(data(:,1), 3*data(:, 104), '--r'); hold on;
plot(data(:,1), 3*data(:, 105), '--g');
plot(data(:,1), 3*data(:, 106), '--b');
plot(data(:,1), 3*data(:, 107), '--k');

plot(data(:,1), -3*data(:, 104), '--r'); hold on;
plot(data(:,1), -3*data(:, 105), '--g');
plot(data(:,1), -3*data(:, 106), '--b');
plot(data(:,1), -3*data(:, 107), '--k');

h_legend= legend('f_x','f_y','c_x','c_y', '3\sigma_f_x', '3\sigma_f_y','3\sigma_c_x','3\sigma_c_y');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('($f_x$, $f_y$), ($c_x$, $c_y$)', 'Interpreter', 'Latex');
set(gca,'FontSize',18);
ylabel('pixel', 'FontSize', 18);

figNumber = figNumber +1;
figure(figNumber);
plot(data(:,1), data(:, 53), '-r'); hold on;
plot(data(:,1), data(:, 54), '-g');
plot(data(:,1), data(:, 55), '-b');
plot(data(:,1), data(:, 56), '-k');

plot(data(:,1), 3*data(:, 108), '--r');
plot(data(:,1), 3*data(:, 109), '--g');
plot(data(:,1), 3*data(:, 110), '--b');
plot(data(:,1), 3*data(:, 111), '--k');

plot(data(:,1), -3*data(:, 108), '--r');
plot(data(:,1), -3*data(:, 109), '--g');
plot(data(:,1), -3*data(:, 110), '--b');
plot(data(:,1), -3*data(:, 111), '--k');

h_legend=legend('k_1','k_2','p_1','p_2', '3\sigma_{k_1}', '3\sigma_{k_2}', '3\sigma_{p_1}', '3\sigma_{p_2}');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('($k_1$, $k_2$, $p_1$, $p_2$)', 'Interpreter', 'Latex');
set(gca,'FontSize',18);


figNumber = figNumber +1;
figure(figNumber);
clock= 1000;
plot(data(:,1), data(:, 57)*clock, '-r'); hold on;
plot(data(:,1), data(:, 58)*clock, '-g');
plot(data(:,1), 3*data(:, 112)*clock, '--r');
plot(data(:,1), 3*data(:, 113)*clock, '--g');
plot(data(:,1), -3*data(:, 112)*clock, '--r');
plot(data(:,1), -3*data(:, 113)*clock, '--g');
h_legend=legend('t_d','t_r', '3\sigma_t_d','3\sigma_t_r');
set(h_legend,'FontSize',18);
grid on;
xlabel('time [sec]', 'FontSize', 18);
title('$t_d$, $t_r$', 'Interpreter', 'Latex');
set(gca,'FontSize',18);
ylabel('msec', 'FontSize', 18);

saveas(gcf,[outputPath, '\Error td tr'],'epsc');

end
