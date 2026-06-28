<%

count = 1

While count <= 10

  icount = 1
  While icount <= 10
    
    If (count Mod 2) = 0 Then
      Response.Write "<"
    Else
      Response.Write ">"      
    End If
 
    icount = icount + 1
  Wend

  Response.Write "<br>"
  count = count + 1
Wend


%>