! simple_model.f90
function simple_add(a, b) result(c)
    implicit none
    real, intent(in) :: a, b
    real :: c
    
    c = a + b
end function simple_add