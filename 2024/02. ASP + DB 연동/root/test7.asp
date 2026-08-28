<%

count = 1

While count <= 10

  icount = 1
  While icount <= 10
    
    If (count=1) Or (count=10) Then
      Response.Write "o"
    Else

      If (icount=1) Or (icount=10) Then
        Response.Write "o"
      Else
        Response.Write "&nbsp;"
      End If
    
    End If

    icount = icount + 1
  Wend

  Response.Write "<br>"
  count = count + 1
Wend


%>