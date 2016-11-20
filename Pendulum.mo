within ;
model Pendulum
  constant Real g=1 "Gravity";

  parameter Real k=150 "Spring Constant";
  parameter Real m=1.0 "Mass";

  Real x(start=1.05);
  Real y(start=0);
  Real vx;
  Real vy;
  Real F;

equation
  der(x) = vx;
  der(y) = vy;
  F = k*(sqrt(x^2 + y^2) -1) / sqrt(x^2 + y^2);
  der(vx) = -(F/1)*x;
  der(vy) = -(F/1)*y-1;

end Pendulum;
