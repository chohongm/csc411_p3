<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  session_start();
  $db = db_connect();
  
  #if user has entered log in info,
  if (isset($_GET['userid'])){
  	$userid = $_GET['userid'];
  	$password = $_GET['password'];
  } else if (isset($_POST['userid'])){
  	$userid = $_POST['userid'];
  	$password = $_POST['password'];
  }
  
  #if the user has yet entered log in info, present log in box.
  if (isset($_GET['url'])){
  	if($_GET['url'] == 'post_form'){
  		$_SESSION['url'] = array('url'=>'post_form', 'parent'=>$_GET['parent']);
  	} else if ($_GET['url'] == 'search_form'){
  		$_SESSION['url'] = array('url'=>'search_form');
  	}
  }
  
  if (!isset($userid) || !isset($password)){
  	do_header();
  	display_login_form();
  	do_footer();
  	
  } else {
	  #query to find a result matching the user's log in info.
	  if($user_info = login($userid, $password)){
	  	#check if user is a web_master.
	  	$m_query = "select * from web_masters where userid = '".$userid."'";
	  	$m_result = mysqli_query($db, $m_query);
	  	$m_row = mysqli_fetch_assoc($m_result);
	  	
	  	if ($m_row != 0){
	  		#user is logged in as master.
	  		$_SESSION['master'] = $userid;
	  	} else if ($user_info['adminid'] != 0){
	  		#user is logged in as admin.
	  		$_SESSION['admin'] = $userid;
	  		#here, admin just logged in for the new day.
	  		$clinicid = admin::get_clinicid($db, $userid);
	  		$clinic = new clinic($db, $clinicid);
	  		/*since the user is an admin starting a new day by logging in, doctors availability
	  		 * should be reset according to their schedule.*/
	  		$clinic->reset_avail_doctors();
	  		#save clinic's info that can be updated and view throughout the day.
	  		$_SESSION['saved_clinic'] = $clinic;
	  		
	  	} else {
	  		#user is logged in as regular user.
	  		$_SESSION['user'] = $userid;
	  	}
	  	
	  	#if user is logged in from other page, go to that page.
	  	if (isset($_SESSION['url'])){
	  		if ($_SESSION['url']['url'] == 'search_form'){
	  			unset($_SERVER['url']);
	  			echo "<meta http-equiv=refresh content=0;URL='/db_entry/search_form.php'>";
	  		} else if ($_SESSION['url']['url'] == 'post_form'){
	  			$parent = $_SESSION['url']['parent'];
	  			unset($_SESSION['url']);
	  			echo "<meta http-equiv=refresh content=0;URL='/forum/post_form.php?parent=$parent'>";
	  		}
	  	} else {
	  		echo "<meta http-equiv=refresh content=0;URL='/pages/home.php'>";
	  	}
	  }
  }
?>