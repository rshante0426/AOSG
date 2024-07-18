function y_new = y2xnorm(y, x)

% Normalizes y data into x data space

y_new = (max(x(:)) - min(x(:))) ./  (max(y(:)) - ...
           min(y(:))) .* (y - min(y(:))) + min(x(:));
end