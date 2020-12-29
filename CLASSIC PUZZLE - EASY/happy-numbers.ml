(* Auto-generated code below aims at helping you parse *)
(* the standard input according to the problem statement. *)

let int_of_char c = int_of_string (Char.escaped c);;

let digits s =
    let l = List.init (String.length s) (String.get s) in
    List.map int_of_char l
;;

let rec sum res array =
    match array with
    | [] -> res
    | hd :: tl -> sum (res + hd) tl
;;

let happy n =
    string_of_int (sum 0 (List.map (fun n -> n*n) (digits n)))
;;

let rec is_happy n set =
    match n with
    | "1" -> ":)"
    | _ ->
        match List.mem n set with
        | true -> ":("
        | false -> is_happy (happy n) (n :: set)

;;

let n = int_of_string (input_line stdin) in
for i = 0 to n - 1 do
    let x = input_line stdin in
    print_endline (x ^ " " ^ (is_happy x []));
    ();
done;
