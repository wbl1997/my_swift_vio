function calculateVelocityInLocalWorldFrame()
% given the rotation matrices, and the velocity in the E-frame obtained by
% GPS/IMU integration, calculate the velocity in the local world frame used
% in okvis and viclam. Then the resultant velocity is used to initialize
% okvis

addpath('/media/jhuai/Seagate/jhuai/JianzhuHuai/GPS_IMU/programs/matlab_ws/voicebox');

close all;

q_w0_s6imu = [	0.695838653 0.0788256255	-0.713859293	0];
R_w0_s6imu = rotqr2ro(q_w0_s6imu');

T_s6imu_s6cam= [  0.000341425919084, -0.999979770238, 0.00635157806524, 0;
           -0.999250868111, -0.000586960849829, -0.0386957110793, 0;
           0.0386986564019, -0.00633360817684, -0.999230853907, 0;
           0.0, 0.0, 0.0, 1.0];
       
R_mic_s6cam =[       -0.996799782290203       0.00294630413299654       -0.0798843746809149;
       -0.0794308700301977        0.0759560869424173         0.993942357354108;
       0.00899616098403661         0.997106810802309       -0.0754789834271593];

q_e_mic =[ 0.029419 0.077296 -0.904587 -0.418190];
R_e_mic =rotqr2ro(q_e_mic');

v_e=[0.139166 4.107111 4.848891]';
v_W0 =R_w0_s6imu*T_s6imu_s6cam(1:3,1:3)*R_mic_s6cam'*R_e_mic'*v_e

end