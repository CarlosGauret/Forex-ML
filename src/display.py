def formatear_direccion(direccion):
    texto = str(direccion)
    if texto == "LONG":
        return "LONG (COMPRAR)"
    if texto == "SHORT":
        return "SHORT (VENDER)"
    return texto


def formatear_texto_direcciones(texto):
    resultado = str(texto)
    resultado = resultado.replace("LONG (COMPRAR)", "__LONG_DISPLAY__")
    resultado = resultado.replace("SHORT (VENDER)", "__SHORT_DISPLAY__")
    resultado = resultado.replace("LONG", "LONG (COMPRAR)")
    resultado = resultado.replace("SHORT", "SHORT (VENDER)")
    resultado = resultado.replace("__LONG_DISPLAY__", "LONG (COMPRAR)")
    resultado = resultado.replace("__SHORT_DISPLAY__", "SHORT (VENDER)")
    return resultado
