clearvars

load('cw1a.mat')
figure(1)
plot(x, y, 'bx')
grid on

csvwrite('cw1a.csv', [x, y])

load('cw1e.mat')
figure(2)
clf
xx0 = reshape(x(:,1),11,11);
xx1 = reshape(x(:,2),11,11);
yy = reshape(y,11,11);
surf(xx0, xx1, yy, 'edgecolor', 'none');

csvwrite('cw1e.csv', [x, y])