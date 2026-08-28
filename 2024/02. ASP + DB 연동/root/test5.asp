<%

oddsum = 0
evensum = 0
count = 1

While count <= 10

  If (count Mod 2) = 0 Then
    evensum = evensum + count
  Else
    oddsum = oddsum + count
  End If

  count = count + 1
Wend

Response.Write evensum
Response.Write "<br>"
Response.Write oddsum

%>