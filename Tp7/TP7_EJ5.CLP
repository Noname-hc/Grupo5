(defrule Evaluar-Enfermedad
    =>
    (printout t "Cual es el estado de la enfermedad?  (Incipiente / Avanzado / Terminal) "crlf) 

    (assert (E_Enfermedad(read)))
)

(defrule Numero_De_Dosis
    (E_Enfermedad ?E_E)
    =>

    (if (eq ?E_E Incipiente)
    then
        
        (bind ?Alto 86)
        (bind ?bajo 75)

        (printout t "Ingrese cantidad de dosis ")
        (bind ?n (read))
        

        (if (>= ?n ?Alto)
        then
            (printout t "Condicion: " crlf)
            (printout t "    -Enfermedad incipiente" crlf)
            (printout t "    -Numero de dosis: Alta (Mayor que 85)" crlf)
            (assert(Efecto Defuncion))

        else
            (if(<= ?n ?bajo)
            then
                (printout t "Condicion: " crlf)
                (printout t "    -Enfermedad incipiente" crlf)
                (printout t "    -Numero de dosis: Baja (Menor que 76)" crlf)
                (assert(Efecto Curacion))
               
            else
                (printout t "Condicion: " crlf)
                (printout t "    -Enfermedad incipiente" crlf)
                (printout t "    -Numero de dosis: Media (entre 76 y 85)" crlf)      
                (assert(Efecto Defuncion))

            )       
        )
    )

)



(defrule estado_Avanzado
    (E_Enfermedad ?E_E)
    =>
    (if(eq ?E_E Avanzado)
    then
        (printout t "Condicion: " crlf)
        (printout t "    -Enfermedad Avanzada" crlf)
        (assert(Efecto Curacion))
    )

)

(defrule Condicion_Fisica 
    (E_Enfermedad ?E_E)
    =>

    (if(eq ?E_E Terminal)
    then
        (printout t "Ingrese la condicion fisica del sujeto (Fuerte / Debil) "crlf)
        (bind ?Cond (read))

        (if(eq ?Cond Fuerte)
        then
            (printout t "Condicion: " crlf)
            (printout t "    -Enfermedad terminal" crlf)
            (printout t "    -Condicion Fisica: Fuerte" crlf)
            (assert(Efecto Defuncion))
        else
            (if(eq ?Cond Debil)
            then 
                (printout t "Condicion: " crlf)
                (printout t "    -Enfermedad terminal" crlf)
                (printout t "    -Condicion Fisica: Debil" crlf)
                (assert(Efecto Curacion))

            else
                (printout t "Condicion ingresada no valida "crlf)
            )

        )
    )
)

(defrule resultado
	(Efecto ?ef)
	(E_Enfermedad ?)
	=>
    (printout t "Resultado: ")
	(printout t ?ef " de la cobaya."crlf)
) 