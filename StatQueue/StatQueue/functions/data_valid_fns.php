<?php

function filled_out($form_vars) {
  // test that each variable has a value
  foreach ($form_vars as $key => $value) {
  	 if ($key != 'clinicid' && $key != 'doctor'){
     	if ((!isset($key)) || ($value == '')) {
          return false;
     	}
     }
  }
  return true;
}

function find_term_like($db, $col, $table, $term){
	if($term == ''){
		return '';
	}
	$query = "select $col from $table where $col like '%".$term."%'";
	$result = $db->query($query);
	$row = $result->fetch_row();
	return $row[0];
}

function valid_account($an){
	$db = db_connect();
	$query = "select userid from authorized_users where userid = '".$an."'";
	$result = $db->query($query);
	if(!preg_match('/^(?=.*[a-zA-Z])[0-9a-zA-Z@\_\.]{6,25}$/', $an) ||
	   $result->num_rows != 0){
		return false;
	}
	return true;
}

function valid_clinicid($ci){
	$db = db_connect();
	$query = "select clinicid from clinics where clinicid = ".$ci;
	$result = $db->query($query);
	if(!preg_match('/^[0-9]+$/', $ci) || $result->num_rows == 0){
		return false;
	}
	return true;
}

function valid_pw($pw){
	/*between start and end of a string, there has to be at least one number,
	 one lowercase, one uppercase letters. As long as these are met, other chars could be any of these or 
	 any of defined special chars. Also the pw length must be btw 6 and 25.*/
	return preg_match('/^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[0-9a-zA-Z!@#$%]{6,25}$/', $pw);
}

function valid_birthdate($bd){
	return preg_match('/^[0-9]{4}\/[0-9]{2}\/[0-9]{2}$/', $bd);
}

function valid_email($email) {
	// check an email address is possibly valid
	return preg_match('/^[a-zA-Z0-9_\.\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+$/', $email);
}

function valid_order($rows, $order1, $order2){
	$num_rows = count($rows);
	if (!in_array($order1, range(1, $num_rows)) || !in_array($order2, range(1, $num_rows))){
		throw new Exception('Given orders are either out of range or order cannot move down.
							 Please try again with correct orders.');
	}
	#return preg_match('/^(?=.*\d)@/', $order1)
	return preg_match('/^(?=.*\d)[0-9]$/', $order2) && preg_match('/^(?=.*\d)[0-9]$/', $order1);
}

function valid_postalcode($postalcode){
	return preg_match('/^[A-Z][0-9][A-Z][0-9][A-Z][0-9]$/', $postalcode);
}

function clean($string) {
	#strip whitespace from the beginning and end of a string.
	$string = trim($string);
	#escaping chars that are special chars in html form.
	$string = htmlentities($string);
	#strip html and php tags from a string.
	$string = strip_tags($string);
	return $string;
}

function clean_all($form_vars) {
	foreach ($form_vars as $key => $value)   {
		$form_vars[$key] = clean($value);
	}
	return $form_vars;
}

?>
