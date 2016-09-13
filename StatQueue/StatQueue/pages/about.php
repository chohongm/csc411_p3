<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();
?>
  <!-- About -->
	<h2 align="center"><br/>Rethinking 'wait-time': You only need to make it on time.<br/><br/></h2>
	<h3>About StatQueue</h3>
	<p> 'StatQueue' is a website that allows any user with valid Canadian
		healthcard to virtually sign-up for waitlists of local walk-in clinics
		or make appointments with a specific doctor. Although the service
		is currently available to users in Toronto only, it may expand nationwide
		in future.</p>
		
  <!-- Purpose -->
    <br/>
	<h3>Why we built it</h3>
	<p> Regardless of exponential improvement in connecting people and
		sharing ideas through information technology in the past decade,
		healthcare service in Canada has not been efficient at utilizing 
		these benefits. Although, being regarded as having one of the best 
		healthcare systems in the world, Canada has consistently been at the 
		bottom of hospital waiting time pool. By creating an information
		technology through which users can virtually sign up for waitlist 
		of walk-in clinics and access information about current waitlist, 
		patients with valid healthcard could signup without being physically
		present and plan their arrival based on estimated
		waiting time. </p>
		
  <!-- How -->
    <br/>
	<h3>Search</h3>
	<p>If you have a valid Canadian healthcard, you are eligible to create an 
	   account at our website. Once logged-in to an account, users have access
	   to statqueue search through which users can acquire a list of currently
	   available clinics located within users' city and simply sign-up for the clinic's
	   waitlist. In addition, in case which users are looking to make an appointment
	   with a specific doctor of clinic, users can search for a doctor, or view 
	   list of doctors of a clinic and make appointment if available.
	   
	   Following waitlist sign-up or appointment made with a doctor,
	   users are directed to 'My Waitlist Status' page where information about
	   users' waitlist queue can be viewed. Here, users have the option to delist
	   themselves from a waitlist if and only if estimated waiting time has yet
	   passed below 30 minutes.
	</p>
		
  <!-- Forum -->
    <br/>
	<h3>Forum</h3>
	<p>If users have personal concern about their health condition, questions
		about the website, clinics, or any general questions, they may wish to post their
		questions on forum page. Any registered user including clinic administrator
		can view and reply to posts. Sharing answers via forum could help others with
		similar concerns. This could potentially reduce number of unnecessary 
		case of patient-doctor visit which in turn would improve the productivity
		of healthcare.</p>
		
  <!-- Questions -->
    <br/>
	<h3>Questions?</h3>
	<p>If you are a clinic administrator and interested in incorporating your clinic to
		StatQueue's virtual waitlist system, contact us via email with information regarding
		your clinic including its clinic ID, full address, as well as your administrator ID.
		These info will be saved to statqueue's database. Shortly after, we will send you a
		reply including required information for you to set up an account as clinic
		administrator. Once you have created your account, you can freely access statqueue's 
		virtual waitlist system for your clinic.<br/>
	<br/><a href="/pages/contact_us.php">Contact Us</a></p>
<?php
  do_footer();
?>