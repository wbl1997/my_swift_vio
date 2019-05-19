function figNumber = intermediatePlotter(... figNumber, data, gt, outputPath,
                                         fontSize) if (nargin <= 3) outputPath =
    "" end if (nargin <= 4) fontSize = 18;
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
clock= 1000;
plot(data(:,1), data(:, 57)*clock, '-r'); hold on;
plot(data(:,1), data(:, 58)*clock, '-g');
plot(data( :, 1), data( :, 57) * clock + 3 * data( :, 112) * clock, '--r');
plot(data( :, 1), data( :, 58) * clock + 3 * data( :, 113) * clock, '--g');
plot(data( :, 1), data( :, 57) * clock - 3 * data( :, 112) * clock, '--r');
plot(data( :, 1), data( :, 58) * clock - 3 * data( :, 113) * clock, '--g');
h_legend=legend('t_d','t_r', '3\sigma_t_d','3\sigma_t_r');
set(h_legend, 'FontSize', fontSize);
grid on;
xlabel('time [sec]', 'FontSize', fontSize);
title('$t_d$, $t_r$', 'Interpreter', 'Latex');
set(gca, 'FontSize', fontSize);
ylabel('msec', 'FontSize', fontSize);

saveas(gcf,[outputPath, '\Error td tr'],'epsc');

end
